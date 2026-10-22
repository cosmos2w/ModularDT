from __future__ import annotations

import json
import random
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from honf_runtime import research_training as research
from honf_runtime.unified_training import EngineConfig, SamplingKey, TaskBatch, TrainingEngine


PROJECT = Path(__file__).resolve().parents[1]
PRESETS = PROJECT / "configs" / "research"


def test_research_presets_keep_membership_and_execution_contracts_separate():
    expected = {
        "thermal_honf_full.json": ("original_train600_fixed25_dev22", 24, 1024, "values"),
        "thermal_dense_full.json": ("original_train600_fixed25_dev22", 24, 1024, "values"),
        "thermal_direct_full.json": ("original_train600_fixed25_dev22", 24, 1024, "values"),
        "thermal_honf_dev150.json": ("fixed25_train150_dev22", 8, 512, "tile"),
        "thermal_dense_dev150.json": ("fixed25_train150_dev22", 8, 512, "tile"),
        "thermal_direct_dev150.json": ("fixed25_train150_dev22", 8, 512, "tile"),
    }
    for name, (membership, microbatch, tile, checkpoint) in expected.items():
        config, _ = research.load_research_config(PRESETS / name)
        assert config["training"]["membership"] == membership
        assert config["execution"]["options"] == {
            "backend": "dense_weighted",
            "query_tile_size": tile,
            "checkpoint": checkpoint,
            "context_backend": "segment",
        }
        assert config["storage"]["output_root"].startswith("/data2/")
        assert config["training"]["effective_batch_size"] == 48
        assert config["training"]["primary_queries"] == 1024
        assert config["training"]["microbatch_cases"] == microbatch

    fresh, _ = research.load_research_config(PRESETS / "thermal_honf_full_fresh_train600_normalization.json")
    assert fresh["training"]["membership"] == "original_train600_fixed25_dev22"
    assert fresh["initialization"]["normalization_mode"] == "fresh_full_train600_stats"
    assert fresh["initialization"]["fresh_model_schedule"]["ramp_epochs"] == 40
    assert fresh["training"]["microbatch_cases"] == 24
    assert fresh["execution"]["options"] == {
        "backend": "dense_weighted",
        "query_tile_size": 1024,
        "checkpoint": "values",
        "context_backend": "segment",
    }
    assert fresh["initialization"]["fresh_controller_lambda0"] == {
        "M": 1.007182002067566,
        "E": 1.59217369556427,
        "context": 1.3138799667358398,
    }
    assert "U10-H" in fresh["initialization"]["fresh_controller_lambda0_source"]
    assert fresh["optimizer"]["schedule"] == "constant_source_terminal_lr"


def test_classic_full5000_profiles_retain_formal_data_and_execution_contracts():
    configs = [research.load_research_config(PRESETS / f"thermal_u_{arm}_full5000_v1.json")[0]
               for arm in "dhs"]
    assert {config["training"]["membership"] for config in configs} == {
        "classic_train600_exposed_val89"
    }
    assert {config["training"]["local_horizon"] for config in configs} == {5000}
    assert {config["training"]["microbatch_cases"] for config in configs} == {24}
    assert {config["training"]["effective_batch_size"] for config in configs} == {48}
    assert {config["training"]["primary_queries"] for config in configs} == {1024}
    assert {config["training"]["seed"] for config in configs} == {0}
    assert {tuple(config["training"]["checkpoint_epochs"]) for config in configs} == {
        (10, 50, 100, 250, 500, 1000, 2500, 5000)
    }
    assert {config["initialization"]["fresh_routing_mode"] for config in configs} == {
        "dense", "hypergraph", "direct"
    }
    assert {config["initialization"]["normalization_mode"] for config in configs} == {
        "classic_train600_stats"
    }
    assert all(research._parent_age(config) == 0 for config in configs)
    assert {research._fresh_model_schedule_epoch(config, 1) for config in configs} == {3001}
    assert {research._fresh_model_schedule_epoch(config, 5000) for config in configs} == {8000}
    for config in configs:
        assert config["execution"]["options"] == {
            "backend": "dense_weighted",
            "query_tile_size": 1024,
            "checkpoint": "values",
            "context_backend": "segment",
        }
        assert config["optimizer"] == {
            "kind": "fresh_adamw_from_classic_formal_contract",
            "schedule": "constant_classic_formal_lr",
        }
        assert config["storage"]["output_root"].startswith(
            "/data2/wanglz/ModularDT/relocated_workspace/HONF_Proj/Trained_Results/ThermalChannel/HONF_Forward_Runs/thermal_u_"
        )


def test_classic_fresh_models_use_metadata_only_and_common_seed0_architecture(monkeypatch):
    from honf_runtime import compat

    def forbid_checkpoint_deserialization(*args, **kwargs):
        pytest.fail("Fresh classic initialization must not deserialize mature checkpoint weights.")

    monkeypatch.setattr(compat, "load_trusted_checkpoint", forbid_checkpoint_deserialization)
    models = {}
    expected_routes = {"d": "dense", "h": "hypergraph", "s": "direct"}
    for arm, route in expected_routes.items():
        config, _ = research.load_research_config(PRESETS / f"thermal_u_{arm}_full5000_v1.json")
        model, source_payload, _, _ = research._source_components(config, torch.device("cpu"))
        assert model.core.routing_mode == route
        assert model.core.get_config()["base_model_config"]["init_seed"] == 0
        assert model.state_payload()["normalizer_key"] == (
            "7ab3441ba2616c68224f9d829a92e2db4c59658b45d3f1acf4c535333896939b"
        )
        assert "model_state_dict" not in source_payload
        assert "optimizer_state_dict" not in source_payload
        assert source_payload["classic_sampling_dataset_id"].startswith("ThermalChannel:")
        models[arm] = model
    states = [{name: value.detach().cpu() for name, value in model.state_dict().items()
               if isinstance(value, torch.Tensor)}
              for model in models.values()]
    common = set(states[0]) & set(states[1]) & set(states[2])
    assert len(common) > 100
    assert all(torch.equal(states[0][name], states[index][name]) for index in (1, 2) for name in common)


def test_sampling_membership_fingerprint_is_shared_across_h_d_s_weights():
    train_ids = ("0001", "0002", "0003")
    dev_ids = ("0277", "0279")
    configs = [research.load_research_config(PRESETS / f"thermal_{arm}_full.json")[0]
               for arm in ("honf", "dense", "direct")]
    assert {config["initialization"]["source_parent_age"] for config in configs} == {5000}
    assert {config["initialization"]["normalization_mode"] for config in configs} == {
        "inherit_source_train150"
    }
    dataset_identity = {
        "dataset_id": "thermal_channel_global_v1",
        "dataset_size_bytes": 1234,
        "dataset_mtime_ns": 5678,
        "catalog_metadata_sha256": "c" * 64,
        "base_manifest_sha256": "d" * 64,
    }
    fingerprints = {
        config["initialization"]["source_arm"]: research.research_membership_fingerprint(
            dataset_identity=dataset_identity,
            training_case_ids=train_ids,
            validation_case_ids=dev_ids,
        )
        for config in configs
    }
    assert len(set(fingerprints.values())) == 1
    assert next(iter(fingerprints.values())) != research.research_membership_fingerprint(
        dataset_identity=dataset_identity,
        training_case_ids=("0001", "0002"),
        validation_case_ids=dev_ids,
    )


def test_fresh_work_ramp_uses_local_sampler_clock_and_native_controller_clock():
    config, _ = research.load_research_config(PRESETS / "thermal_honf_full_fresh_train600_normalization.json")
    schedule = config["initialization"]["fresh_model_schedule"]
    assert research._fresh_model_schedule_epoch(config, 1) == 3025
    assert research._fresh_model_schedule_epoch(config, 40) == 4000
    engine = research.ResearchTrainingEngine(
        EngineConfig(seed=0, microbatch_cases=2, effective_cases=2, total_epochs=1000,
                     training_mode="joint"),
        device="cpu",
        parent_age=0,
        fresh_model_schedule=schedule,
    )
    assert engine.sampler_epoch(1) == 1
    assert engine.sampler_epoch(40) == 40
    contract = engine.sampler_epoch_contract()
    assert contract["parent_age"] == 0
    assert contract["fresh_model_schedule"] == schedule
    assert "coordinate_start_epoch" in contract["fit_epoch_to_model_clock"]


def test_fresh_controller_configuration_rejects_disabled_work_penalties():
    config, _ = research.load_research_config(
        PRESETS / "thermal_honf_full_fresh_train600_normalization.json"
    )
    config["initialization"]["fresh_controller_lambda0"]["E"] = 0.0
    with pytest.raises(ValueError, match="positive calibrated strengths"):
        research._validate_research_config(config)


class _TinyDual:
    def __init__(self, lambda0: float):
        self.lambda0 = torch.tensor(lambda0)
        self.dual_lambda = torch.tensor(lambda0)
        self.zeta = torch.tensor(lambda0)
        self.lambda_max = torch.tensor(10 * lambda0)
        self.eta_lambda = torch.tensor(lambda0)
        self.macro_step = torch.tensor(23)
        self.window_step_count = torch.tensor(4)
        self.dual_update_count = torch.tensor(7)
        self.window_residual_sum = torch.tensor(2.5)

    def set_calibrated_lambda0(self, value):
        self.lambda0.fill_(value)
        self.dual_lambda.fill_(value)
        self.zeta.fill_(value)
        self.lambda_max.fill_(10 * value)
        self.eta_lambda.fill_(value)


class _TinyBudget:
    def schedule(self):
        return {"absolute_epoch": 3000, "sparse_age": 0}

    def __init__(self):
        self.duals = {name: _TinyDual(9.0) for name in ("M", "E", "context")}


class _TinyCore(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([weight], dtype=torch.float32))
        self.joint_epoch = 3000
        self.joint_macro_clock = 0
        self.joint_budget = _TinyBudget()
        self.execution_options = None

    def get_config(self):
        return {"architecture": "tiny"}

    @classmethod
    def from_config(cls, config):
        assert config == {"architecture": "tiny"}
        return cls(weight=1.0)

    def set_execution_options(self, **options):
        self.execution_options = options

    def set_training_clock(self, epoch, update_index):
        self.joint_epoch = int(epoch)
        self.joint_macro_clock = int(update_index)


def test_fresh_core_uses_config_initialization_and_resets_native_state():
    source = _TinyCore(weight=99.0)
    fresh = research._fresh_core_from_template(
        source, seed=17,
        execution_options={"backend": "dense_weighted", "query_tile_size": 512,
                           "checkpoint": "tile", "context_backend": "segment"},
        controller_lambda0={"M": 1.0, "E": 1.5, "context": 1.25},
    )
    assert fresh is not source
    assert float(fresh.weight.item()) == 1.0
    assert float(source.weight.item()) == 99.0
    assert fresh.joint_epoch == 3000
    assert fresh.joint_macro_clock == 0
    assert fresh.execution_options["backend"] == "dense_weighted"
    for name, expected in {"M": 1.0, "E": 1.5, "context": 1.25}.items():
        dual = fresh.joint_budget.duals[name]
        assert float(dual.lambda0) == expected
        assert float(dual.dual_lambda) == expected
        assert float(dual.zeta) == expected
        assert float(dual.lambda_max) == 10 * expected
        assert float(dual.eta_lambda) == expected
        assert int(dual.macro_step) == 0
        assert int(dual.window_step_count) == 0
        assert int(dual.dual_update_count) == 0
        assert float(dual.window_residual_sum) == 0.0


def test_run_manifest_is_self_contained_after_initial_source_is_gone(tmp_path):
    config, _ = research.load_research_config(PRESETS / "thermal_honf_full.json")
    config["initialization"]["source_checkpoint"] = str(tmp_path / "removed-parent.pt")
    config_path = tmp_path / "removed-config.json"
    run = tmp_path / "run"
    run.mkdir()
    research._write_run_manifest(
        run,
        config=config,
        config_path=config_path,
        source_path=Path(config["initialization"]["source_checkpoint"]),
        source_sha="a" * 64,
        optimizer_contract=[{"name": "source_read_joint", "parameter_names": [], "schedule": {}}],
        case_ids=("0001", "0002"),
        storage_snapshot={},
        diagnostic=True,
        batch=2,
        microbatch=2,
        identity_microbatch=2,
    )
    loaded, saved_config_path = research._load_run_manifest(run)
    assert saved_config_path == config_path
    assert loaded["source_checkpoint"] == str(config_path.parent / "removed-parent.pt")
    assert loaded["config"]["initialization"]["source_checkpoint"].endswith("removed-parent.pt")
    assert not Path(loaded["source_checkpoint"]).exists()


def test_run_checkpoint_rejects_foreign_paths_and_mismatched_identity(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    identity = {"family": research.TASK_SCHEMA, "config_name": "thermal_u_h_full5000_v1"}
    (run / "experiment_identity.json").write_text(json.dumps(identity))
    checkpoint = run / "latest_model.pt"
    torch.save({"experiment_identity": identity, "epoch": 10}, checkpoint)
    payload, selected = research._read_run_checkpoint(run)
    assert payload["epoch"] == 10 and selected == checkpoint
    foreign = tmp_path / "foreign.pt"
    torch.save({"experiment_identity": identity, "epoch": 10}, foreign)
    with pytest.raises(ValueError, match="outside the declared research run"):
        research._read_run_checkpoint(run, str(foreign))
    torch.save({"experiment_identity": {**identity, "config_name": "thermal_u_d_full5000_v1"}}, checkpoint)
    with pytest.raises(ValueError, match="identity differs"):
        research._read_run_checkpoint(run)


class _RetryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3.0]))


class _InjectedOOMEngine:
    def __init__(self, model, initial_rng, *, fail):
        self.model = model
        self.initial_rng = initial_rng
        self.fail = fail
        self.device = torch.device("cpu")
        self.parent_age = 5
        self.config = SimpleNamespace(microbatch_cases=2, effective_cases=2)

    def fit(self, model, provider, output, **kwargs):
        del kwargs
        if self.fail:
            with torch.no_grad():
                model.weight.add_(10)
            provider.mutable_state += 1
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError("CUDA out of memory: injected retry-boundary check")
        assert torch.equal(model.weight.detach(), torch.tensor([3.0]))
        assert provider.mutable_state == 0
        state = research.capture_rng_state()
        assert state["python"] == self.initial_rng["python"]
        assert state["numpy"][0] == self.initial_rng["numpy"][0]
        assert np.array_equal(state["numpy"][1], self.initial_rng["numpy"][1])
        assert torch.equal(state["torch_cpu"], self.initial_rng["torch_cpu"])
        return {"completed_epoch": 1, "status": "injected_retry_passed"}


def test_first_epoch_oom_retries_from_model_and_all_rng_boundary(tmp_path, monkeypatch):
    previous_rng = research.capture_rng_state()
    try:
        random.seed(73)
        np.random.seed(73)
        torch.manual_seed(73)
        expected_rng = research.capture_rng_state()
        model = _RetryModel()
        provider = SimpleNamespace(
            device=torch.device("cpu"),
            microbatch_size=2,
            _identity_microbatch_size=2,
            _research_case_limit=2,
            _research_config={"training": {"microbatch_cases": 2}},
            mutable_state=0,
        )
        provider.training_state_dict = lambda: {"mutable_state": provider.mutable_state}
        provider.load_training_state_dict = lambda state: setattr(provider, "mutable_state", state["mutable_state"])
        output = tmp_path / "retry-run"
        output.mkdir()
        (output / research._CONFIG_NAME).write_text(json.dumps({"microbatch_cases": 2}))
        first = _InjectedOOMEngine(model, expected_rng, fail=True)
        second = _InjectedOOMEngine(model, expected_rng, fail=False)
        engines = iter((second,))
        monkeypatch.setattr(research, "_engine_for", lambda *args, **kwargs: next(engines))
        result = research._fit_with_retries(
            first, model, provider, output,
            identity={}, stop_after=1, arm="A2-D",
        )
        assert result["status"] == "injected_retry_passed"
        assert (tmp_path / "retry-run.oom_attempt_01" / "oom_retry_01.json").is_file()
    finally:
        research.restore_rng_state(previous_rng)


def test_outer_oom_downshift_is_not_overwritten_by_initial_cli_microbatch(tmp_path, monkeypatch):
    options = {
        "backend": "dense_weighted",
        "query_tile_size": 1024,
        "checkpoint": "values",
        "context_backend": "segment",
    }
    config = EngineConfig(
        seed=4, microbatch_cases=8, effective_cases=48, total_epochs=10,
        training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION,
    )

    class _OutOfMemoryEngine(research.ResearchTrainingEngine):
        def fit(self, model, provider, output, **kwargs):
            del model, provider, kwargs
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            (output / "latest_model.pt").write_bytes(b"sealed latest checkpoint marker")
            raise RuntimeError("CUDA out of memory: injected outer-retry boundary")

    class _RetryEngine(research.ResearchTrainingEngine):
        def fit(self, model, provider, output, **kwargs):
            del model, kwargs
            assert self.nominal_microbatch_cases == 8
            assert self.runtime_microbatch_cases == 12
            assert self.config.microbatch_cases == 8
            assert provider.microbatch_size == 12
            runtime = json.loads((Path(output) / "execution_runtime.json").read_text())
            assert runtime["nominal_microbatch_cases"] == 8
            assert runtime["runtime_microbatch_cases"] == 12
            assert runtime["runtime_execution_options"] == options
            return {"status": "outer_oom_downshift_retained"}

    initial = _OutOfMemoryEngine(config, device="cpu", parent_age=5000, execution_options=options)
    retry = _RetryEngine(config, device="cpu", parent_age=5000, execution_options=options)
    model = _RetryModel()
    provider = SimpleNamespace(
        device=torch.device("cpu"),
        microbatch_size=8,
        _identity_microbatch_size=8,
        _research_config={},
        _research_case_limit=None,
    )
    monkeypatch.setattr(research, "_engine_for", lambda *args, **kwargs: retry)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = research._fit_with_retries(
        initial, model, provider, tmp_path / "outer-retry",
        identity={}, stop_after=5, arm="A2-H",
        runtime_microbatch_cases=24,
        runtime_execution_options=options,
    )
    assert result["status"] == "outer_oom_downshift_retained"


def test_local_oom_retry_restores_the_last_complete_optimizer_macro(tmp_path, monkeypatch):
    import fcntl

    previous_rng = research.capture_rng_state()
    try:
        random.seed(109)
        np.random.seed(109)
        torch.manual_seed(109)
        model = _RetryModel()
        provider = SimpleNamespace(
            sampling_dataset_id="thermal-membership-test",
            mutable_state=0,
            query_draws={"case-a": [], "case-b": []},
        )

        def make_batch(case_chunk, key):
            for case_id in case_chunk:
                provider.query_draws[case_id].append(tuple(map(int, key.native_indices(
                    1000, 8, case_id, "fluid", "primary",
                ))))
            return TaskBatch(None, None, None, {"sampling_key": key}, tuple(case_chunk))

        def training_state_dict():
            return {"mutable_state": provider.mutable_state}

        def load_training_state_dict(state):
            provider.mutable_state = int(state["mutable_state"])

        provider.make_batch = make_batch
        provider.training_state_dict = training_state_dict
        provider.load_training_state_dict = load_training_state_dict
        provider.loss_denominators = lambda microbatches, phase, arm: {
            "field": sum(len(batch.case_keys) for batch in microbatches),
        }
        engine = research.ResearchTrainingEngine(
            EngineConfig(seed=13, microbatch_cases=2, effective_cases=2, total_epochs=4,
                         training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION),
            device="cpu", parent_age=5,
        )
        output = tmp_path / "local-retry"
        output.mkdir()
        engine.bind_execution_runtime(output / "execution_runtime.json", nominal_microbatch_cases=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        order = ("case-a", "case-b")
        first_batches = engine.prepare_update_microbatches(
            provider, arm="A2-H", phase="joint", epoch=1, update_index=0,
            update_cases=order, sampling_dataset_id=provider.sampling_dataset_id,
        )

        def optimizer_snapshot():
            return research._cpu_state_snapshot(optimizer.state_dict())

        def assert_nested_equal(left, right):
            if torch.is_tensor(left):
                assert torch.equal(left, right)
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    assert_nested_equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                assert len(left) == len(right)
                for left_item, right_item in zip(left, right):
                    assert_nested_equal(left_item, right_item)
            else:
                assert left == right

        first_macro_weight = None
        first_macro_optimizer = None
        failed_entry_rng = None
        seen_cases = []
        second_macro_calls = 0

        def injected_parent_update(self, active_model, active_provider, active_optimizer, microbatches,
                                   *, phase, arm, epoch, update_index, **kwargs):
            nonlocal first_macro_weight, first_macro_optimizer, failed_entry_rng, second_macro_calls
            del self, phase, arm, epoch, kwargs
            cases = tuple(case for batch in microbatches for case in batch.case_keys)
            seen_cases.append(cases)
            denominator = active_provider.loss_denominators(microbatches, "joint", "A2-H")["field"]
            if update_index == 1:
                second_macro_calls += 1
                if second_macro_calls == 1:
                    assert torch.equal(active_model.weight.detach(), first_macro_weight)
                    assert_nested_equal(optimizer_snapshot(), first_macro_optimizer)
                    assert active_provider.mutable_state == 1
                    failed_entry_rng = research._cpu_state_snapshot(research.capture_rng_state())
                else:
                    assert cases == order
                    assert torch.equal(active_model.weight.detach(), first_macro_weight)
                    assert_nested_equal(optimizer_snapshot(), first_macro_optimizer)
                    assert active_provider.mutable_state == 1
                    current_rng = research.capture_rng_state()
                    assert current_rng["python"] == failed_entry_rng["python"]
                    assert current_rng["numpy"][0] == failed_entry_rng["numpy"][0]
                    assert np.array_equal(current_rng["numpy"][1], failed_entry_rng["numpy"][1])
                    assert torch.equal(current_rng["torch_cpu"], failed_entry_rng["torch_cpu"])
            active_optimizer.zero_grad(set_to_none=True)
            loss = sum(
                active_model.weight.square().sum() * (1.0 if case == "case-a" else 2.0)
                for case in cases
            ) / denominator
            loss.backward()
            active_optimizer.step()
            active_provider.mutable_state += 1
            if update_index == 0:
                first_macro_weight = active_model.weight.detach().clone()
                first_macro_optimizer = optimizer_snapshot()
            elif second_macro_calls == 1:
                with torch.no_grad():
                    active_model.weight.add_(17.0)
                random.random()
                np.random.random()
                torch.rand(4)
                raise RuntimeError("CUDA out of memory: injected partial second-macro update")
            return {}, {}, {"field": float(denominator)}, None

        monkeypatch.setattr(TrainingEngine, "_run_update", injected_parent_update)
        lock_path = output / ".training.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            engine._run_update(
                model, provider, optimizer, first_batches,
                phase="joint", arm="A2-H", epoch=1, update_index=0,
            )
            second_batches = engine.prepare_update_microbatches(
                provider, arm="A2-H", phase="joint", epoch=1, update_index=1,
                update_cases=order, sampling_dataset_id=provider.sampling_dataset_id,
            )
            engine._run_update(
                model, provider, optimizer, second_batches,
                phase="joint", arm="A2-H", epoch=1, update_index=1,
            )
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        runtime = json.loads((output / "execution_runtime.json").read_text())
        assert engine.config.microbatch_cases == 2
        assert engine.nominal_microbatch_cases == 2
        assert engine.runtime_microbatch_cases == 1
        assert engine.local_oom_retry_count == 1
        assert runtime["nominal_microbatch_cases"] == 2
        assert runtime["runtime_microbatch_cases"] == 1
        assert runtime["local_oom_retry_count"] == 1
        assert runtime["last_oom"]["status"] == "repacked_same_optimizer_macro"
        assert runtime["last_oom"]["ordered_case_keys"] == list(order)
        resumed_engine = research.ResearchTrainingEngine(
            EngineConfig(seed=13, microbatch_cases=2, effective_cases=2, total_epochs=4,
                         training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION),
            device="cpu", parent_age=5,
        )
        resumed_engine.bind_execution_runtime(
            output / "execution_runtime.json", nominal_microbatch_cases=2,
        )
        assert resumed_engine.config.microbatch_cases == 2
        assert resumed_engine.runtime_microbatch_cases == 1
        assert resumed_engine.local_oom_retry_count == 1
        mismatched_engine = research.ResearchTrainingEngine(
            EngineConfig(seed=13, microbatch_cases=1, effective_cases=2, total_epochs=4,
                         training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION),
            device="cpu", parent_age=5,
        )
        with pytest.raises(ValueError, match="nominal microbatch identity"):
            mismatched_engine.bind_execution_runtime(
                output / "execution_runtime.json", nominal_microbatch_cases=1,
            )
        assert seen_cases == [order, order, order]
        assert all(len(set(draws)) == 1 for draws in provider.query_draws.values())
        assert int(optimizer.state[model.weight]["step"]) == 2
        assert provider.mutable_state == 2
    finally:
        research.restore_rng_state(previous_rng)


def test_local_oom_exhaustion_does_not_enter_epoch_checkpoint_fallback(tmp_path, monkeypatch):
    output = tmp_path / "local-oom-exhausted"
    output.mkdir()
    model = _RetryModel()
    provider = SimpleNamespace(
        _identity_microbatch_size=2,
        training_state_dict=lambda: {},
    )

    class _ExhaustedEngine:
        device = torch.device("cpu")
        config = SimpleNamespace(microbatch_cases=2, effective_cases=2)

        def fit(self, *args, **kwargs):
            del args, kwargs
            raise research.ResearchLocalOOMExhausted("local macro CUDA OOM retries exhausted")

    monkeypatch.setattr(
        research, "_engine_for",
        lambda *args, **kwargs: pytest.fail("Local update exhaustion must not rewind from an epoch checkpoint."),
    )
    with pytest.raises(research.ResearchLocalOOMExhausted):
        research._fit_with_retries(
            _ExhaustedEngine(), model, provider, output,
            identity={}, stop_after=1, arm="A2-H",
        )
    assert not list(output.glob("oom_retry_*.json"))


def test_clean_stop_handler_writes_a_resumable_request(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    previous = research._install_clean_stop_handlers(output)
    try:
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
    finally:
        research._restore_handlers(previous)
    request = json.loads((output / "CLEAN_STOP_REQUEST.json").read_text())
    assert request["signal"] == int(signal.SIGINT)
    assert request["request_id"]


def test_resume_stop_supports_an_absolute_total_epoch_target():
    assert research._resolve_resume_stop(
        current_epoch=120, local_horizon=600, additional_epochs=None, stop_after_epoch=200,
    ) == 200
    assert research._resolve_resume_stop(
        current_epoch=120, local_horizon=600, additional_epochs=8, stop_after_epoch=None,
    ) == 128
    with pytest.raises(ValueError, match="Specify exactly one"):
        research._resolve_resume_stop(
            current_epoch=120, local_horizon=600, additional_epochs=8, stop_after_epoch=200,
        )
    with pytest.raises(ValueError, match="greater than the current checkpoint"):
        research._resolve_resume_stop(
            current_epoch=200, local_horizon=600, additional_epochs=None, stop_after_epoch=200,
        )


def test_runtime_pack_larger_than_nominal_preserves_identity_and_query_keys(tmp_path):
    options = {
        "backend": "dense_weighted",
        "query_tile_size": 512,
        "checkpoint": "tile",
        "context_backend": "segment",
    }
    engine_config = EngineConfig(
        seed=31, microbatch_cases=8, effective_cases=48, total_epochs=10,
        training_mode="joint", sampling_version=SamplingKey.CASE_EPOCH_VERSION,
    )
    runtime_path = tmp_path / "execution_runtime.json"
    engine = research.ResearchTrainingEngine(
        engine_config, device="cpu", parent_age=5000, execution_options=options,
    )
    engine.bind_execution_runtime(
        runtime_path, nominal_microbatch_cases=8, default_execution_options=options,
    )

    case_order = tuple(f"thermal-{index:03d}" for index in range(48))

    class _BatchProvider:
        def __init__(self):
            self.microbatch_size = 8
            self._identity_microbatch_size = 8
            self.draws = {}

        def make_batch(self, cases, key):
            assert len(cases) <= self.microbatch_size
            for case_id in cases:
                self.draws[case_id] = tuple(map(int, key.native_indices(
                    128, 16, case_id, "fluid", "primary",
                )))
            return TaskBatch(None, None, None, {"sampling_key": key}, tuple(cases))

        def identity_payload(self):
            current_options = self.core.execution_options
            payload = {
                "microbatch_size": self._identity_microbatch_size,
                "model_config": {"execution_options": current_options},
                "adapter_state": {"core_config": {"execution_options": current_options}},
            }
            return research._restore_checkpoint_identity_metadata(self, payload)

    identity_core = _TinyCore(weight=1.0)
    identity_core.set_execution_options(**options)
    runtime_provider = _BatchProvider()
    runtime_provider.core = identity_core
    runtime_provider._research_identity_model_config = {"execution_options": options}
    runtime_provider._research_identity_adapter_state = {
        "core_config": {"execution_options": options},
    }
    identity_before = research._canonical_sha256(runtime_provider.identity_payload())
    runtime_options = {**options, "query_tile_size": 1024, "checkpoint": "values"}
    research._apply_runtime_execution_options(identity_core, runtime_provider, runtime_options)
    engine.set_runtime_overrides(
        provider=runtime_provider, microbatch_cases=24, execution_options=runtime_options,
    )
    runtime_batches = engine.prepare_update_microbatches(
        runtime_provider, arm="A2-H", phase="joint", epoch=3, update_index=11,
        update_cases=case_order, sampling_dataset_id="thermal-fixed-membership",
    )
    identity_after = research._canonical_sha256(runtime_provider.identity_payload())
    assert identity_after == identity_before
    assert identity_core.execution_options == runtime_options
    assert [len(batch.case_keys) for batch in runtime_batches] == [24, 24]
    assert tuple(case for batch in runtime_batches for case in batch.case_keys) == case_order
    runtime_draws = dict(runtime_provider.draws)

    nominal_engine = research.ResearchTrainingEngine(
        engine_config, device="cpu", parent_age=5000, execution_options=options,
    )
    nominal_provider = _BatchProvider()
    nominal_batches = nominal_engine.prepare_update_microbatches(
        nominal_provider, arm="A2-H", phase="joint", epoch=3, update_index=11,
        update_cases=case_order, sampling_dataset_id="thermal-fixed-membership",
    )
    assert [len(batch.case_keys) for batch in nominal_batches] == [8, 8, 8, 8, 8, 8]
    assert tuple(case for batch in nominal_batches for case in batch.case_keys) == case_order
    assert nominal_provider.draws == runtime_draws
    assert sum(len(batch.case_keys) for batch in runtime_batches) == 48
    assert sum(len(batch.case_keys) for batch in nominal_batches) == 48

    engine._execution_runtime_path = runtime_path
    engine._write_execution_runtime(last_oom={"status": "execution_override_test"})
    saved = json.loads(runtime_path.read_text())
    assert saved["nominal_microbatch_cases"] == 8
    assert saved["runtime_microbatch_cases"] == 24
    assert saved["runtime_execution_options"] == runtime_options
    loaded_options = research._execution_options_for_run(
        tmp_path,
        {"execution": {"options": options}},
    )
    assert loaded_options == runtime_options

    resumed_engine = research.ResearchTrainingEngine(
        engine_config, device="cpu", parent_age=5000, execution_options=options,
    )
    resumed_engine.bind_execution_runtime(
        runtime_path, nominal_microbatch_cases=8, default_execution_options=options,
    )
    resumed_provider = _BatchProvider()
    resumed_engine.set_runtime_overrides(provider=resumed_provider)
    assert resumed_engine.config.microbatch_cases == 8
    assert resumed_engine.nominal_microbatch_cases == 8
    assert resumed_engine.runtime_microbatch_cases == 24
    assert resumed_provider.microbatch_size == 24
    assert resumed_engine.runtime_execution_options == runtime_options
    mismatched_engine = research.ResearchTrainingEngine(
        engine_config, device="cpu", parent_age=5000, execution_options=options,
    )
    with pytest.raises(ValueError, match="nominal microbatch identity"):
        mismatched_engine.bind_execution_runtime(
            runtime_path, nominal_microbatch_cases=16, default_execution_options=options,
        )
