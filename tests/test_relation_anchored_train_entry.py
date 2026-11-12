"""Supported case dispatch and saved execution identities."""
import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "relation_anchored_train", Path(__file__).resolve().parents[1] / "tools/relation_anchored_train.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


@pytest.mark.parametrize("device,fraction", [("cpu", ".44"), ("cuda:0", "0"),
                                            ("cuda:0", "1.1"), ("cuda:0", "nan")])
def test_cuda_memory_limit_rejects_invalid_device_or_fraction(monkeypatch, device, fraction):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid allocator arguments must not initialize CUDA or build a model")

    monkeypatch.setattr(cli.torch.cuda, "set_per_process_memory_fraction", unexpected)
    monkeypatch.setattr(cli, "build", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main(["inspect", "--recipe", "unused.json", "--device", device,
                  "--cuda-memory-fraction", fraction])
    assert error.value.code == 2


def test_cuda_memory_limit_preserves_science_and_targets_selected_device(monkeypatch):
    recipe = {"dataset": {"name": "thermal"}, "member_routing": {"arm": "joint"},
              "training": {"microbatch_cases": 24, "primary_queries": 1024}}
    monkeypatch.setattr(Path, "read_text", lambda self: json.dumps(recipe))
    events = []
    monkeypatch.setattr(cli.torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, *, device: events.append((fraction, device)))

    def build(actual_recipe, *, device):
        assert events == [(.44, "cuda:1")]
        assert actual_recipe == recipe and device == "cuda:1"
        return object(), object()

    class Engine:
        def fit(self, model, provider, output, **kwargs):
            assert kwargs["identity"]["recipe"] == recipe
            assert kwargs["resume_checkpoint"] == "latest" and kwargs["stop_after"] == 1000
            return {"completed_epoch": 1000}

    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "engine_for", lambda *args, **kwargs: Engine())
    assert cli.main(["fit", "--recipe", "same.json", "--device", "cuda:1",
                     "--output", "same", "--resume", "latest", "--stop-after", "1000",
                     "--cuda-memory-fraction", ".44"]) == 0


def wind_recipe():
    return {"dataset": "wind", "seed": 42, "architecture": "relation_anchored",
            "schema": "wind_relation_anchored_v1", "microbatch_size": 8,
            "effective_batch_size": 24, "total_epochs": 100}


def test_horizon_and_sampling_do_not_depend_on_physical_packing():
    class Cases:
        def identity_payload(self):
            return {"dataset": "WindFarm", "subset_manifest_sha256": "same-membership"}

        def epoch_cases(self, epoch, seed):
            return tuple(range(72))

    recipe = wind_recipe()
    one = cli.engine_for(recipe, device="cpu")
    other = cli.engine_for({**recipe, "microbatch_size": 4}, device="cpu")
    assert one.epoch_case_order(Cases(), 1) == other.epoch_case_order(Cases(), 1)
    assert one.config.total_epochs == 100
    assert one.config.checkpoint_epochs == (100,)
    assert one.config.sampling_version == "case_epoch_v1"
    assert one.config.curve_every == 10


def test_explicit_extension_preserves_recipe_and_rejects_other_scopes():
    recipe = wind_recipe()
    before = copy.deepcopy(recipe)
    engine = cli.engine_for(recipe, device="cpu", execution_horizon=300)
    assert recipe == before
    assert engine.config.checkpoint_epochs == (100, 200, 300)
    for horizon in (100, 99, 301):
        with pytest.raises(ValueError):
            cli.engine_for(recipe, device="cpu", execution_horizon=horizon)
    with pytest.raises(ValueError):
        cli.engine_for({**recipe, "schema": "wind_relation_anchored_r15_3_v1"},
                       device="cpu", execution_horizon=300)


def test_p64_dispatch_and_excluded_experiments(monkeypatch):
    import sys

    module = ModuleType("channelthermal.training.thermal_port_response_task")
    calls = {}
    expected = (object(), object())

    def build(recipe, *, arm, device):
        calls.update(arm=arm, device=device)
        return expected

    module.build_thermal_port_response_task = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    recipe = {"dataset": {"name": "thermal"}, "initialization": {"seed": 0},
              "member_routing": {"arm": "R14-Detail-LocalECover"},
              "thermal_port_response": {"enabled": True, "arm": "P64"}}
    assert cli.build(recipe, device="cpu") == expected
    assert calls == {"arm": "R14-Detail-LocalECover", "device": "cpu"}
    for name in ("thermal_feedback", "thermal_finite_response", "local_expert"):
        with pytest.raises(ValueError, match="research branch"):
            cli.build({**recipe, name: {"enabled": True}}, device="cpu")


def test_scratch_family_dispatch_receives_literal_resume_state(monkeypatch):
    import sys

    module = ModuleType('channelthermal.training.r16_10_family_task')
    expected = (object(), object())
    saved = {'epoch': 10, 'provider_training_state': {'cursor': 0}}
    recipe = {'dataset': {'name': 'thermal'}, 'initialization': {'seed': 0},
              'member_routing': {'arm': 'R14-Detail-LocalECover'},
              'r16_10_family': {'arm': 'R'}}

    def build(actual, *, arm, device, resume_payload):
        assert actual is recipe and resume_payload is saved
        assert arm == 'R14-Detail-LocalECover' and device == 'cpu'
        return expected

    module.build_thermal_r16_10_task = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    health = ModuleType('channelthermal.training.r16_10_health')
    attached = []
    health.attach_update_diagnostics = lambda provider, actual: attached.append((provider, actual))
    monkeypatch.setitem(sys.modules, health.__name__, health)
    assert cli.build(recipe, device='cpu', resume_payload=saved) == expected
    assert attached == []  # Campaign diagnostics stay on the preserved research branch.


def test_posttrain_evaluation_dispatch_uses_child_state(monkeypatch):
    import sys

    module = ModuleType('channelthermal.training.r16_10_posttrain')
    recipe = {'dataset': {'name': 'thermal'}, 'initialization': {'seed': 0},
              'member_routing': {'arm': 'R14-Detail-LocalECover'},
              'r16_10_family': {'arm': 'R'}, 'r16_10_posttrain': {'source_epoch': 3000}}
    saved = {'epoch': 100}
    expected = (object(), object())
    def build(actual, *, device, source, resume_payload):
        assert actual is recipe and device == 'cpu' and source is None and resume_payload is saved
        return expected
    module.build_posttrain_task = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert cli.build(recipe, device='cpu', resume_payload=saved) == expected


def test_posttrain_fit_requires_dedicated_state_preserving_entry(monkeypatch):
    recipe = {'r16_10_posttrain': {'source_epoch': 3000}}
    monkeypatch.setattr(Path, 'read_text', lambda self: json.dumps(recipe))
    with pytest.raises(SystemExit) as error:
        cli.main(['fit', '--recipe', 'post.json', '--output', 'child', '--stop-after', '1000'])
    assert error.value.code == 2


def test_scratch_family_resume_loads_saved_state_without_legacy_scales(monkeypatch):
    recipe = {'dataset': {'name': 'thermal'}, 'member_routing': {'arm': 'joint'},
              'r16_10_family': {'arm': 'B0'}}
    saved = {'epoch': 10, 'experiment_identity': {'recipe': copy.deepcopy(recipe)}}
    monkeypatch.setattr(Path, 'read_text', lambda self: json.dumps(recipe))
    monkeypatch.setattr(cli, 'resolve_checkpoint', lambda *args: Path('/unused'))
    monkeypatch.setattr(cli, 'load_trusted_checkpoint', lambda *args, **kwargs: saved)

    def build(actual, *, device, resume_payload):
        assert actual == recipe and resume_payload is saved
        return object(), object()

    class Engine:
        def fit(self, model, provider, output, **kwargs):
            assert kwargs['resume_checkpoint'] == 'latest'
            assert kwargs['stop_after'] == 1000
            return {'completed_epoch': 1000}

    monkeypatch.setattr(cli, 'build', build)
    monkeypatch.setattr(cli, 'engine_for', lambda *args, **kwargs: Engine())
    assert cli.main(['fit', '--recipe', 'run.json', '--output', 'run', '--resume', 'latest',
                     '--stop-after', '1000']) == 0
    saved['experiment_identity']['recipe']['r16_10_family']['arm'] = 'R'
    with pytest.raises(ValueError, match='recipe differs'):
        cli.main(['fit', '--recipe', 'run.json', '--output', 'run', '--resume', 'latest',
                  '--stop-after', '1000'])


def test_fit_preserves_same_run_resume_arguments(monkeypatch, capsys):
    recipe = wind_recipe()
    monkeypatch.setattr(Path, "read_text", lambda self: json.dumps(recipe))
    model = object()
    provider = SimpleNamespace(preparation_summary=lambda: {"catalogue_build_count": 0})
    monkeypatch.setattr(cli, "build", lambda *args, **kwargs: (model, provider))
    calls = {}

    class Engine:
        def fit(self, actual_model, actual_provider, output, **kwargs):
            calls.update(model=actual_model, provider=actual_provider, output=output, **kwargs)
            return {"completed_epoch": 100}

    monkeypatch.setattr(cli, "engine_for", lambda *args, **kwargs: Engine())
    assert cli.main(["fit", "--recipe", "run.json", "--output", "run",
                     "--resume", "latest", "--stop-after", "100"]) == 0
    assert calls["model"] is model and calls["provider"] is provider
    assert calls["identity"]["recipe"] == recipe
    assert calls["resume_checkpoint"] == "latest" and calls["stop_after"] == 100
    preparation, result = map(json.loads, capsys.readouterr().out.splitlines())
    assert preparation["preparation"]["catalogue_build_count"] == 0
    assert result["completed_epoch"] == 100


def test_invalid_extension_is_rejected_before_loading_inputs(monkeypatch):
    monkeypatch.setattr(cli, "build", lambda *args, **kwargs: pytest.fail("invalid command built inputs"))
    with pytest.raises(SystemExit):
        cli.main(["fit", "--recipe", "run.json", "--output", "run", "--stop-after", "300", "--extend-horizon"])


@pytest.mark.parametrize('coefficient', [None, 0.])
def test_response_resume_restores_saved_scales_and_keeps_exact_recipe(monkeypatch, coefficient):
    response = {'enabled': True, 'schema': 'thermal_state_response_v1'}
    if coefficient is not None:
        response['finite_response_coefficient'] = coefficient
    recipe = {'dataset': {'name': 'thermal'}, 'member_routing': {'arm': 'joint'},
              'thermal_state_response': response}
    scales = {'flow/u': {'scale': .45560681823737065, 'fit_rms': .45560681823737065, 'floor': .01}}
    checkpoint = {'experiment_identity': {'recipe': copy.deepcopy(recipe),
                                           'provider_identity': {'response_scales': scales,
                                               'adapter_state': {'normalization_stats': {'saved': [1.0]}},
                                               'source_binding': {'normalization_source': 'saved TRAIN stats'}}}}
    monkeypatch.setattr(Path, 'read_text', lambda self: json.dumps(recipe))
    monkeypatch.setattr(cli, 'resolve_checkpoint', lambda *args: Path('/unused'))
    monkeypatch.setattr(cli, 'load_trusted_checkpoint', lambda *args, **kwargs: checkpoint)
    captured = {}

    def build(actual_recipe, *, device, response_scales, saved_adapter_state, saved_normalization_source):
        assert saved_adapter_state == checkpoint['experiment_identity']['provider_identity']['adapter_state']
        assert saved_normalization_source == 'saved TRAIN stats'
        captured.update(recipe=actual_recipe, scales=response_scales, device=device)
        return object(), object()

    class Engine:
        def fit(self, model, provider, output, **kwargs):
            assert kwargs['identity']['recipe'] == recipe
            assert kwargs['resume_checkpoint'] == 'latest' and kwargs['stop_after'] == 200
            return {'completed_epoch': 200}

    monkeypatch.setattr(cli, 'build', build)
    monkeypatch.setattr(cli, 'engine_for', lambda *args, **kwargs: Engine())
    assert cli.main(['fit', '--recipe', 'run.json', '--output', 'run', '--resume', 'latest',
                     '--stop-after', '200']) == 0
    assert captured == {'recipe': recipe, 'scales': scales, 'device': 'cpu'}
    checkpoint['experiment_identity']['recipe']['thermal_state_response']['finite_response_coefficient'] = .2
    with pytest.raises(ValueError, match='recipe differs'):
        cli.main(['fit', '--recipe', 'run.json', '--output', 'run', '--resume', 'latest',
                  '--stop-after', '200'])


def test_r16_10_extension_accepts_latest_only_if_saved_epoch_is_2000(monkeypatch):
    recipe = {'dataset': {'name': 'thermal'}, 'member_routing': {'arm': 'joint'},
              'r16_10_family': {'arm': 'B0'}}
    saved = {'epoch': 2000, 'experiment_identity': {'recipe': copy.deepcopy(recipe),
                                                    'engine_config': {'total_epochs': 2000}}}
    monkeypatch.setattr(Path, 'read_text', lambda self: json.dumps(recipe))
    monkeypatch.setattr(cli, 'resolve_checkpoint', lambda *args: Path('/unused/latest.pt'))
    monkeypatch.setattr(cli, 'load_trusted_checkpoint', lambda *args, **kwargs: saved)
    monkeypatch.setattr(cli, 'build', lambda *args, **kwargs: (object(), object()))
    class Engine:
        def fit(self, model, provider, output, **kwargs):
            assert kwargs['resume_checkpoint'] == 'latest'
            assert kwargs['allow_horizon_extension'] is True
            return {'completed_epoch': 3000}
    monkeypatch.setattr(cli, 'engine_for', lambda *args, **kwargs: Engine())
    arguments = ['fit', '--recipe', 'run.json', '--output', 'run', '--resume', 'latest',
                 '--extend-horizon', '--stop-after', '3000']
    assert cli.main(arguments) == 0
    saved['epoch'] = 1900
    with pytest.raises(ValueError, match='completed epoch 2000 state'):
        cli.main(arguments)
