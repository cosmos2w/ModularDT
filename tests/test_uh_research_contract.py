"""The retained U-H research path preserves its in-run resume contract."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from honf_runtime import research_training as research


PROJECT = Path(__file__).resolve().parents[1]


def _config():
    return research.load_research_config(PROJECT / "configs/research/thermal_u_h_full5000_v1.json")[0]


@pytest.mark.parametrize("change", ["route", "normalization", "source", "round12"])
def test_research_validator_rejects_unselected_model_families(change):
    config = copy.deepcopy(_config())
    if change == "route":
        config["initialization"]["fresh_routing_mode"] = "dense"
    elif change == "normalization":
        config["initialization"]["normalization_mode"] = "inherit_parent_train_stats"
    elif change == "source":
        config["initialization"]["source_kind"] = "scratch_compact_config"
    else:
        config["round12"] = {}
    with pytest.raises(ValueError, match="retained U-H"):
        research._validate_research_config(config)


def test_mocked_next_epoch_uses_existing_checkpoint_and_contract(tmp_path, monkeypatch):
    config = _config()
    manifest = {"config": config, "optimizer_schedule_contract": [{"name": "saved"}],
                "source_checkpoint": "/saved/source.pt", "source_checkpoint_sha256": "a" * 64}
    payload = {"epoch": 100, "experiment_identity": {"engine_config": {"total_epochs": 5000}}}
    checkpoint = tmp_path / "checkpoints" / "latest_model.pt"
    model = SimpleNamespace(architecture="A2-D")
    provider = SimpleNamespace(_identity_microbatch_size=24, effective_batch_size=48,
                               training_case_ids=("case",))
    engine = SimpleNamespace()
    calls = {}
    monkeypatch.setattr(research, "_load_run_manifest", lambda run: (manifest, run))
    monkeypatch.setattr(research, "resolve_device", lambda *a, **k: (torch.device("cpu"), {}))
    monkeypatch.setattr(research, "verify_storage", lambda *a, **k: {})
    monkeypatch.setattr(research, "_read_run_checkpoint", lambda *a: (payload, checkpoint))
    monkeypatch.setattr(research, "_execution_options_for_run", lambda *a: config["execution"]["options"])
    monkeypatch.setattr(research, "_runtime_microbatch_from_file", lambda *a: 24)
    def build(*args, **kwargs):
        calls["provider_args"] = args
        return model, provider
    monkeypatch.setattr(research, "_build_provider_for_resume", build)
    monkeypatch.setattr(research, "_engine_for", lambda *a, **k: engine)
    monkeypatch.setattr(research, "_extend_execution_horizon", lambda *a, **k: None)
    monkeypatch.setattr(research, "_experiment_identity", lambda *a, **k: {"saved": "identity"})
    def fit(*args, **kwargs):
        calls["fit"] = kwargs
        return {"mock": "ready"}
    monkeypatch.setattr(research, "_fit_with_retries", fit)
    assert research.resume(tmp_path, device="cpu", stop_after_epoch=101) == {"mock": "ready"}
    assert calls["provider_args"][1] is payload
    assert calls["provider_args"][2] == manifest["optimizer_schedule_contract"]
    assert calls["fit"]["resume_checkpoint"] == checkpoint
    assert calls["fit"]["stop_after"] == 101
    assert calls["fit"]["identity"] == {"saved": "identity"}
