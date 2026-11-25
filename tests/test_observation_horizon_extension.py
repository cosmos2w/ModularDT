"""Explicit observation-response horizon changes keep saved science and state."""

import copy
import json
from types import SimpleNamespace

import pytest

from honf_runtime.unified_training import _engine_config_payload, _resume_identity_amendment
from tools import relation_anchored_train as cli


def _recipe(dataset=None):
    return {
        "dataset": dataset or {"name": "thermal"},
        "training_profile": "observation_response",
        "observation_response": {"mode": "coupled_native", "response_weight": 1.0},
        "training": {"seed": 3, "local_horizon": 500, "optimizer_schedule_horizon": 2000,
                     "microbatch_cases": 24, "effective_batch_size": 48,
                     "checkpoint_epochs": [100, 200, 300, 400, 500], "curve_every": 10},
        "optimizer": {"lr": 0.0003, "schedule": "constant_config_lr"},
    }


def _identity(recipe):
    return {
        "workflow": "relation_anchored_case_training_v1",
        "recipe": copy.deepcopy(recipe),
        "engine_config": _engine_config_payload(cli.engine_for(recipe, device="cpu").config),
        "provider_identity": {"calibration": 0.8, "membership": [1, 2], "training": {"total_epochs": 500}},
        "optimizer_schedule_contract": [{"name": "joint", "schedule": {
            "peak_lr": 0.0003, "warmup_start_lr": 0.0003, "warmup_epochs": 0,
            "hold_through_epoch": 2000, "total_epochs": 2001, "final_lr": 0.0003}}],
    }


def _extended(identity, horizon=1000):
    result = copy.deepcopy(identity)
    result["engine_config"] = _engine_config_payload(
        cli.engine_for(identity["recipe"], device="cpu", execution_horizon=horizon).config)
    return result


def test_observation_extension_preserves_schedule_and_prior_milestones_for_any_case():
    for dataset in ({"name": "thermal"}, {"name": "another_case"}):
        recipe = _recipe(dataset)
        recipe["training"]["checkpoint_epochs"] = [50, 300, 500]
        old = _identity(recipe)
        new = _extended(old)
        assert new["engine_config"]["checkpoint_epochs"] == (50, 300, 500, 600, 700, 800, 900, 1000)
        assert old["recipe"]["training"]["local_horizon"] == 500
        assert old["optimizer_schedule_contract"] == new["optimizer_schedule_contract"]
        with pytest.raises(ValueError):
            _resume_identity_amendment(old, new)
        amendment = _resume_identity_amendment(old, new, allow_horizon_extension=True)
        assert amendment["kind"] == "explicit_observation_response_horizon_extension"
        assert amendment["optimizer_schedule_unchanged"]
        later = _extended(old, 1200)
        assert _resume_identity_amendment(new, later, allow_horizon_extension=True)["previous_horizon"] == 1000


@pytest.mark.parametrize("path,value", [
    (("recipe", "training", "local_horizon"), 1000),
    (("recipe", "training", "seed"), 4),
    (("recipe", "observation_response", "response_weight"), 0.5),
    (("provider_identity", "calibration"), 0.7),
    (("provider_identity", "membership"), [1, 3]),
    (("provider_identity", "training", "total_epochs"), 1000),
    (("optimizer_schedule_contract", 0, "schedule", "total_epochs"), 1001),
    (("engine_config", "microbatch_cases"), 12),
    (("engine_config", "checkpoint_epochs"), [100, 200, 300, 400, 600, 700, 800, 900, 1000]),
])
def test_observation_extension_rejects_other_identity_changes(path, value):
    old = _identity(_recipe())
    changed = _extended(old)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        _resume_identity_amendment(old, changed, allow_horizon_extension=True)


def test_observation_cli_retains_extended_horizon_on_ordinary_resume(tmp_path, monkeypatch):
    recipe = _recipe()
    old = _identity(recipe)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, sort_keys=True))
    checkpoint = {"epoch": 500, "experiment_identity": old}
    provider = SimpleNamespace()
    model = SimpleNamespace(architecture="joint")
    calls = []
    monkeypatch.setattr(cli, "resolve_checkpoint", lambda output, selector: tmp_path / ("latest.pt" if selector == "latest" else "other.pt"))
    monkeypatch.setattr(cli, "load_trusted_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(cli, "build", lambda recipe, *, device, resume_payload: (model, provider))
    monkeypatch.setattr(cli, "bind_training_budget", lambda *args: None)
    monkeypatch.setattr(cli, "prepare_profile_fit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.TrainingEngine, "fit", lambda engine, *args, **kwargs:
                        calls.append((engine.config, kwargs)) or {"completed_epoch": checkpoint["epoch"]})
    args = ["fit", "--recipe", str(recipe_path), "--output", str(tmp_path / "run"),
            "--device", "cpu", "--resume", "latest", "--stop-after", "1000"]
    with pytest.raises(ValueError, match="requires --extend-horizon"):
        cli.main(args)
    assert calls == []
    assert cli.main(args + ["--extend-horizon"]) == 0
    config, options = calls[-1]
    assert config.total_epochs == provider._round6_sampling_epoch_limit == 1000
    assert config.checkpoint_epochs == tuple(range(100, 1001, 100))
    assert options["identity"]["recipe"] == recipe and options["allow_horizon_extension"]
    checkpoint.update(epoch=600, experiment_identity=_extended(old))
    assert cli.main(args[:-1] + ["800"]) == 0
    config, options = calls[-1]
    assert config.total_epochs == provider._round6_sampling_epoch_limit == 1000
    assert config.checkpoint_epochs == tuple(range(100, 1001, 100))
    assert not options["allow_horizon_extension"] and options["stop_after"] == 800
    with pytest.raises(ValueError, match="requires --extend-horizon"):
        cli.main(args[:-1] + ["1100"])
    with pytest.raises(ValueError, match="latest checkpoint"):
        cli.main(args[:-3] + ["other", "--stop-after", "1200", "--extend-horizon"])


def test_thermal_sampler_accepts_extended_epoch_without_changing_native_identity():
    from channelthermal.training.content_hypergraph_task import ThermalContentHypergraphTask
    from honf_runtime.unified_training import SamplingKey

    class ReachedCaseLookup(Exception):
        pass

    class CaseLookup:
        def __getitem__(self, key):
            raise ReachedCaseLookup

    provider = SimpleNamespace(sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                               sampling_dataset_id="saved", total_epochs=500,
                               microbatch_size=24, _train_index_by_id=CaseLookup())
    key = SamplingKey(seed=3, epoch=501, update_index=0, microbatch_index=0,
                      stage="joint", arm="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION,
                      dataset_id="saved")
    with pytest.raises(ValueError, match="invalid epoch"):
        ThermalContentHypergraphTask.make_batch(provider, ["case"], key)
    provider._round6_sampling_epoch_limit = 1000
    with pytest.raises(ReachedCaseLookup):
        ThermalContentHypergraphTask.make_batch(provider, ["case"], key)
    assert provider.total_epochs == 500
