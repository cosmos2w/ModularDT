from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.run_layout import RunLayout, resolve_checkpoint
from test_unified_training import _ToyProvider, _config


class _BoundedProvider(_ToyProvider):
    def __init__(self, diagnostic=None):
        super().__init__()
        self.diagnostic = diagnostic

    def identity_payload(self):
        result = {**super().identity_payload(), "train_case_ids": [0, 1, 2, 3, 4]}
        if self.diagnostic is not None:
            result["diagnostic"] = self.diagnostic
        return result

    def epoch_cases(self, epoch, seed):
        return (0, 1) if self.diagnostic is not None else super().epoch_cases(epoch, seed)


def _declaration(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"selection_partition": "TRAIN", "WindTEST_targets": "locked",
                                    "solver_attempts": 0, "allowed_loss_weight_overrides": [{}]}))
    return {"kind": "reference_anchored_objective_diagnostic", "source_epoch": 3,
            "source_checkpoint_sha256": "a" * 64, "review_stop": 5, "review_epochs": [4, 5],
            "fit_case_keys": [0, 1], "scientific_update_limit": 2,
            "loss_weight_overrides": {},
            "evidence_path": str(evidence), "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "reason": "Fixed TRAIN objective counterfactual."}


def _identities():
    parent = {"provider_identity": _BoundedProvider().identity_payload(),
              "engine_config": runtime._engine_config_payload(_config()),
              "optimizer_schedule_contract": [{"group": "fixed"}],
              "selection_policy": asdict(runtime.SelectionPolicy())}
    child = copy.deepcopy(parent)
    child["provider_identity"]["diagnostic"] = {"fit_case_keys": [0, 1], "loss_weight_overrides": {}}
    return parent, child


def test_bounded_age_declaration_retains_parent_exposure_and_counts_child_updates(tmp_path):
    parent, child = _identities()
    result = runtime._diagnostic_age_branch_amendment(
        parent, child, _declaration(tmp_path), source_sha256="a" * 64, case_keys=(1, 0), stop_after=4)
    assert result["updates_per_epoch"] == 1
    assert result["planned_optimizer_updates"] == 2
    assert result["parent_train_exposure_retained"] is True


@pytest.mark.parametrize("change", ["provider", "schedule", "sampler", "case", "budget", "evidence", "stop",
                                    "initialization", "nested_control", "override", "unselected_override"])
def test_bounded_age_declaration_rejects_contract_drift_before_stepping(tmp_path, change):
    parent, child = _identities()
    declaration = _declaration(tmp_path)
    stop = 4
    cases = (0, 1)
    if change == "provider":
        child["provider_identity"]["manifest"] = "resampled"
    elif change == "schedule":
        child["optimizer_schedule_contract"] = [{"group": "changed"}]
    elif change == "sampler":
        child["engine_config"]["seed"] += 1
    elif change == "case":
        cases = (0, 2)
    elif change == "budget":
        declaration["scientific_update_limit"] = 1
    elif change == "evidence":
        declaration["evidence_sha256"] = "b" * 64
    elif change == "initialization":
        child["initial_model_state_sha256"] = "changed"
    elif change == "nested_control":
        child["provider_identity"]["diagnostic"]["new_sampler"] = True
    elif change == "override":
        child["provider_identity"]["diagnostic"]["loss_weight_overrides"] = {"loss": 0.0}
    elif change == "unselected_override":
        declaration["loss_weight_overrides"] = {"loss": 0.0}
        child["provider_identity"]["diagnostic"]["loss_weight_overrides"] = {"loss": 0.0}
    else:
        stop = 6
    with pytest.raises(ValueError):
        runtime._diagnostic_age_branch_amendment(
            parent, child, declaration, source_sha256="a" * 64, case_keys=cases, stop_after=stop)


def test_age_branch_and_exact_resume_keep_moments_and_stop_at_declared_budget(tmp_path, monkeypatch):
    """Two disposable optimizer updates in total; parent moments are fixtures."""
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    config = replace(_config(), monitor_epochs=(4, 5), checkpoint_epochs=(4, 5))
    engine = runtime.TrainingEngine(config, device="cpu")
    parent_provider = _BoundedProvider()
    model = nn.Linear(1, 1)
    optimizer, specs = engine._make_optimizer(model, parent_provider, "full_detail", 3, None)
    for parameter in model.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(3.0),
                                      "exp_avg": torch.full_like(parameter, 0.01),
                                      "exp_avg_sq": torch.full_like(parameter, 0.1)}
    identity = {"run": "parent", "provider_identity": parent_provider.identity_payload(),
                "engine_config": runtime._engine_config_payload(_config()),
                "selection_policy": asdict(runtime.SelectionPolicy()),
                "optimizer_schedule_contract": [runtime._optimizer_spec_payload(spec) for spec in specs]}
    payload = engine._checkpoint_payload(
        model, optimizer, identity=identity, arm="full_detail", epoch=3,
        history=[{"epoch": 3, "case_visits": 20, "optimizer_updates": 7}],
        sampler_state={}, best_field=1.0, best_guarded=1.0, group_specs=specs)
    source = tmp_path / "parent.pt"
    torch.save(payload, source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    declaration = _declaration(tmp_path)
    declaration["source_checkpoint_sha256"] = source_hash
    provider = _BoundedProvider({"fit_case_keys": [0, 1], "loss_weight_overrides": {}})
    source_loader = runtime._load_named_optimizer_state
    loaded_moments = []

    def observe_restored_moments(candidate, candidate_optimizer, seed):
        source_loader(candidate, candidate_optimizer, seed)
        if seed is not None:
            loaded_moments.append({name: copy.deepcopy(candidate_optimizer.state[parameter])
                                   for name, parameter in candidate.named_parameters()})

    monkeypatch.setattr(runtime, "_load_named_optimizer_state", observe_restored_moments)
    child = nn.Linear(1, 1)
    run = tmp_path / "child"
    first = engine.fit(child, provider, run, identity={"run": "diagnostic"}, arm="full_detail",
                       stop_after=4, branch_from_checkpoint=source, diagnostic_age_branch=declaration)
    assert first["completed_epoch"] == 4
    assert all(torch.equal(state["exp_avg"], torch.full_like(state["exp_avg"], 0.01))
               and torch.equal(state["exp_avg_sq"], torch.full_like(state["exp_avg_sq"], 0.1))
               for state in loaded_moments[0].values())
    assert json.loads(RunLayout(run).read_path("fit_summary.json").read_text())["optimizer_updates"] == 1
    resumed = nn.Linear(1, 1)
    final = engine.fit(resumed, provider, run, identity={"run": "diagnostic"}, arm="full_detail",
                       stop_after=5, resume_checkpoint=resolve_checkpoint(run, "latest"))
    assert final["completed_epoch"] == 5
    assert json.loads(RunLayout(run).read_path("fit_summary.json").read_text())["cumulative_optimizer_updates"] == 2
    saved = torch.load(resolve_checkpoint(run, "latest"), weights_only=False)
    assert [row["epoch"] for row in saved["history"]] == [4, 5]
    assert saved["branch_parent"]["inherited_optimizer_updates"] == 7
    assert all(int(state["step"].item()) == 5
               for state in saved["optimizer_state_by_name"]["state_by_name"].values())
    # Source second moments retain at least the two-step .999 decay even
    # without new gradient contributions; allow one FP32 rounding margin.
    assert all(bool(torch.all(state["exp_avg_sq"] >= 0.09979))
               for state in saved["optimizer_state_by_name"]["state_by_name"].values())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    with pytest.raises(ValueError, match="declared reviews or update budget"):
        engine.fit(nn.Linear(1, 1), provider, run, identity={"run": "diagnostic"}, arm="full_detail",
                   stop_after=6, resume_checkpoint=resolve_checkpoint(run, "latest"))
