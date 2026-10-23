"""Fresh compact initialization and automatic, checkpointed algorithm boundaries."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from honf_runtime import research_training as research
from honf_runtime.unified_training import capture_rng_state
from tests.test_research_training_runner import PRESETS, _round12_compact_adapter, _round12_migration_identity
from tests.test_thermal_content_hypergraph_v2 import _core_config


def _config():
    return research.load_research_config(PRESETS / "thermal_compact_tiestop_full1000_scratch_v1.json")[0]


@pytest.mark.parametrize("section,key,value", [
    ("initialization", "source_checkpoint", "/old/weights.pt"),
    ("initialization", "source_parent_age", 1000),
    ("initialization", "fresh_model_schedule", {
        "coordinate_start_epoch": 3000, "coordinate_span_epochs": 1000, "ramp_epochs": 40}),
    ("round12", "tie_loss_gradient", "original"),
    ("round12", "cut_after_local_epoch", 400),
    ("training", "checkpoint_epochs", list(range(100, 1001, 100))),
    ("training", "membership", "fixed25_train150_dev22"),
])
def test_scratch_recipe_rejects_old_weights_or_changed_frozen_contract(section, key, value):
    config = _config()
    config[section][key] = value
    with pytest.raises(ValueError):
        research._validate_research_config(config)


def test_scratch_components_are_random_deterministic_and_never_load_a_checkpoint(monkeypatch):
    from channelthermal.training import content_hypergraph_task_v2 as task

    template = _round12_compact_adapter()
    state = template.state_payload()
    # The maintained factory supplies fresh statistics and architecture only.
    base = SimpleNamespace(
        training_case_ids=tuple(map(str, range(600))), validation_case_ids=tuple(map(str, range(89))),
        model=SimpleNamespace(state_payload=lambda: state,
                              core=SimpleNamespace(get_config=lambda: {**_core_config("A2-D"), "context_depth": 2})),
        identity_payload=lambda: {"adapter_state": state},
    )
    monkeypatch.setattr(task, "build_thermal_content_hypergraph_task_v2", lambda *a, **k: (None, base, None))
    def forbidden(*a, **k):
        raise AssertionError("Scratch initialization must not load any checkpoint")
    monkeypatch.setattr(torch, "load", forbidden)
    config = _config()
    first, payload, path, _ = research._source_components(config, torch.device("cpu"))
    torch.manual_seed(987)
    second, _, _, _ = research._source_components(config, torch.device("cpu"))
    assert path.suffix == ".json" and payload["pretrained_weights_loaded"] is False
    assert int(first.core.compact_local_epoch) == 0 and not bool(first.core.compact_cut_active)
    assert int(first.core.joint_macro_clock) == 0 and int(first.core.joint_epoch) == 3000
    assert first.core.compact_parameter_migration["new_names"]
    for name, parameter in first.named_parameters():
        torch.testing.assert_close(parameter, dict(second.named_parameters())[name], rtol=0, atol=0)
    assert payload["experiment_identity"]["optimizer_schedule_contract"][0]["schedule"]["final_lr"] == 3e-4


def test_scratch_calibration_uses_own_warmup_once_and_survives_retry(tmp_path, monkeypatch):
    model = _round12_compact_adapter()
    provider = SimpleNamespace(_research_config=_config())
    calls = []
    monkeypatch.setattr(research, "_read_run_checkpoint", lambda *a: ({"epoch": 20}, None))
    def calibrate(run, payload, config, device):
        calls.append(payload["epoch"])
        return {"record": {"source_common_local_epoch": 20,
                           "terms": {"tie": {"coefficient": .1}, "capacity": {"coefficient": .2}}},
                "coefficients": {"tie": .1, "capacity": .2}}
    monkeypatch.setattr(research, "_round12_calibrate_structural_losses", calibrate)
    for _ in range(2):
        research._scratch_round12_boundary(tmp_path, model, provider, local_epoch=21,
                                           update_index=0, device=torch.device("cpu"))
    assert calls == [20]
    assert float(model.core.compact_tie_loss_coefficient) == pytest.approx(.1)
    assert float(model.core.compact_capacity_loss_coefficient) == pytest.approx(.2)


def test_scratch_cut_keeps_pre_and_post_states_without_advancing_adam_rng_or_sampler(tmp_path):
    config = _config()
    model = _round12_compact_adapter()
    model.core.set_compact_phase(40, arm="merge")
    identity = _round12_migration_identity(model, config, "scratch-sampling-id")
    layout = research.RunLayout(tmp_path)
    layout.ensure()
    research._atomic_json(layout.write_path("experiment_identity.json"), identity)
    payload = {
        "epoch": 40, "arm": "A2-D", "experiment_identity": identity,
        "model_state_dict": research._cpu_state_snapshot(model.state_dict()),
        "provider_training_state": {"compact_state": model.core.export_compact_state()},
        "optimizer_state_by_name": {"moment": torch.tensor([8.])},
        "rng_state": capture_rng_state(), "sampler_state": {"epoch": 40, "cursor": 600},
    }
    torch.save(payload, layout.write_path("epoch_0040_model.pt"))
    torch.save(payload, layout.write_path("latest_model.pt"))
    provider = SimpleNamespace(_research_config=config)
    research._scratch_round12_boundary(tmp_path, model, provider, local_epoch=41,
                                       update_index=0, device=torch.device("cpu"))
    before, _ = research._read_run_checkpoint(tmp_path, "epoch_0040_model.pt")
    after, _ = research._read_run_checkpoint(tmp_path, "epoch_0040_after_cut_model.pt")
    latest, _ = research._read_run_checkpoint(tmp_path, "latest")
    assert not before["provider_training_state"]["compact_state"]["cut_active"]
    assert after["provider_training_state"]["compact_state"]["cut_active"]
    assert after["epoch"] == before["epoch"] == 40
    assert after["optimizer_state_by_name"] == before["optimizer_state_by_name"]
    assert after["sampler_state"] == before["sampler_state"]
    assert after["rng_state"]["torch_cpu"].equal(before["rng_state"]["torch_cpu"])
    assert latest["provider_training_state"]["compact_state"] == after["provider_training_state"]["compact_state"]
    frozen = research._round12_model_parameter_names(model)
    assert frozen and all(not p.requires_grad for n, p in model.named_parameters() if n in frozen)
    restored = research._model_from_run_payload(after, torch.device("cpu"))
    assert bool(restored.core.compact_cut_active)
    # Re-entry after a recovered update must not overwrite the declared boundary.
    research._scratch_round12_boundary(tmp_path, model, provider, local_epoch=41,
                                       update_index=0, device=torch.device("cpu"))
