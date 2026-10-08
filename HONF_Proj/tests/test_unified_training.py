from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    SelectionPolicy,
    TaskBatch,
    TrainingEngine,
)


class _ToyProvider:
    def __init__(self, *, bad_prepass: bool = False):
        self.bad_prepass = bad_prepass
        self.validation_phases = []
        self.training_mode_seen = []

    def identity_payload(self):
        return {"dataset": "toy", "manifest": "fixed"}

    def epoch_cases(self, epoch: int, seed: int):
        return tuple(range(5))

    def make_batch(self, case_keys, key: SamplingKey):
        x = torch.tensor(case_keys, dtype=torch.float32).reshape(-1, 1)
        y = 2.0 * x + 1.0
        stream_hash = hashlib.sha256(json.dumps(
            [key.seed_for("toy_queries"), list(case_keys)], separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        return TaskBatch(scene_inputs=x, receivers=None, targets=y,
                         auxiliary={"query_sampling_sha256": stream_hash}, case_keys=tuple(case_keys))

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        count = sum(batch.targets.numel() for batch in batches)
        return {"native": float(count + (1 if self.bad_prepass else 0))}

    def make_scene(self, scene_inputs):
        # This callback only receives the explicitly whitelisted input tensor.
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        if not model.training:
            self.validation_phases.append((phase, execution_mode, epoch, temperature))
        else:
            self.training_mode_seen.append(model.training)
        return model(scene), None

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        numerator = (predictions - targets).square().sum()
        return {"native": LossTerm(numerator, targets.numel())}

    def validation_batches(self):
        for keys in ((0, 1, 2), (3, 4)):
            yield self.make_batch(keys, SamplingKey(0, 1, 0, 0, "warmup", "warmup"))

    def validation_metrics(self, predictions, targets, auxiliary_state):
        return {"squared_error": float((predictions - targets).square().sum()), "count": targets.numel()}

    def reduce_native_metrics(self, records):
        total = sum(row["squared_error"] for row in records)
        count = sum(row["count"] for row in records)
        return {"field_score": total / count}

    def optimizer_groups(self, model, arm, stage):
        return (OptimizerGroupSpec(
            name="predictor",
            parameter_names=("weight", "bias"),
            schedule=ScheduleSpec(peak_lr=0.02, warmup_start_lr=0.01, warmup_epochs=2,
                                 hold_through_epoch=4, total_epochs=10, final_lr=0.001),
            weight_decay=0.0,
        ),)

    def on_phase_start(self, *, model, arm, epoch, phase, temperature):
        del arm, epoch, phase, temperature
        # Simulate a provider preflight/calibration callback that leaves the
        # model in eval mode. The engine must reassert training mode.
        model.eval()

    def work_counts(self, batch, predictions, auxiliary_state):
        return {"query_rows": batch.targets.numel()}


class _GuardedToyProvider(_ToyProvider):
    def reduce_native_metrics(self, records):
        return {**super().reduce_native_metrics(records), "response_guard_max_ratio": 0.9}


def _config():
    return EngineConfig(seed=17, microbatch_cases=2, effective_cases=4, total_epochs=10,
                        warmup_epochs=2, open_through_epoch=4, soft_through_epoch=6,
                        monitor_every=1, gradient_clip=10.0)


def _sparse_config():
    return replace(_config(), monitor_every=4, monitor_epochs=(2,),
                   checkpoint_epochs=(2, 6, 10), latest_every=2, curve_every=2)


def test_sparse_outputs_keep_only_milestones_and_update_real_curves(tmp_path):
    provider = _ToyProvider()
    output = tmp_path / "sparse"
    engine = TrainingEngine(_sparse_config(), device="cpu")
    engine.fit(nn.Linear(1, 1), provider, output, identity={"run": "sparse"},
               arm="adaptive_detail", stop_after=10)
    history = json.loads((output / "history.json").read_text())
    assert [row["epoch"] for row in history if "validation" in row] == [2, 4, 8, 10]
    assert {path.name for path in output.glob("epoch_*_model.pt")} == {
        "epoch_0002_model.pt", "epoch_0006_model.pt", "epoch_0010_model.pt"}
    for name in ("loss_curves.pdf", "loss_curves.png", "best_by_field_mse_model.pt"):
        assert (output / name).stat().st_size > 0
    progress = json.loads((output / "progress.json").read_text())
    assert progress["completed_epoch"] == progress["latest_checkpoint_epoch"] == 10
    assert progress["completed_case_visits"] == 50
    assert progress["completed_optimizer_updates"] == 20
    assert not (output / "last.pt").is_symlink()
    assert (output / "last.pt").read_bytes() == (output / "latest_model.pt").read_bytes()
    assert torch.load(output / "last.pt", map_location="cpu", weights_only=False)["epoch"] == 10
    summary = json.loads((output / "fit_summary.json").read_text())
    assert summary["cumulative_microbatches"] == 30
    assert summary["cumulative_partial_update_cases"] == 10


def test_curve_rng_is_preserved_and_sparse_resume_is_exact(tmp_path, monkeypatch):
    class StochasticProvider(_ToyProvider):
        def predict_native(self, model, *args, **kwargs):
            prediction, auxiliary = super().predict_native(model, *args, **kwargs)
            if model.training:
                prediction = prediction + torch.rand_like(prediction) * 0.05 + np.random.rand() * 0.05
            return prediction, auxiliary

    def noisy_curve(*args):
        torch.rand(7)
        np.random.rand(7)

    monkeypatch.setattr(runtime, "_render_loss_curves", noisy_curve)
    torch.manual_seed(88)
    initial = nn.Linear(1, 1).state_dict()
    engine = TrainingEngine(_sparse_config(), device="cpu")
    uninterrupted = nn.Linear(1, 1)
    uninterrupted.load_state_dict(initial)
    torch.manual_seed(91)
    np.random.seed(91)
    engine.fit(uninterrupted, StochasticProvider(), tmp_path / "full_sparse",
               identity={"run": "sparse"}, arm="adaptive_detail", stop_after=10)
    expected_rng = torch.get_rng_state().clone()
    expected_numpy_rng = np.random.get_state()
    interrupted = nn.Linear(1, 1)
    interrupted.load_state_dict(initial)
    output = tmp_path / "resume_sparse"
    torch.manual_seed(91)
    np.random.seed(91)
    engine.fit(interrupted, StochasticProvider(), output,
               identity={"run": "sparse"}, arm="adaptive_detail", stop_after=4)
    torch.rand(9)
    np.random.rand(9)
    engine.fit(interrupted, StochasticProvider(), output, identity={"run": "sparse"},
               arm="adaptive_detail", stop_after=10, resume_checkpoint=output / "latest_model.pt")
    assert torch.equal(torch.get_rng_state(), expected_rng)
    actual_numpy_rng = np.random.get_state()
    assert np.array_equal(actual_numpy_rng[1], expected_numpy_rng[1])
    assert actual_numpy_rng[2:] == expected_numpy_rng[2:]
    for name, value in uninterrupted.state_dict().items():
        assert torch.equal(value, interrupted.state_dict()[name])
    summary = json.loads((output / "fit_summary.json").read_text())
    assert summary["case_visits"] == 30
    assert summary["cumulative_case_visits"] == 50


def test_sparse_nonmilestone_clean_stop_saves_latest_without_extra_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args: None)
    output = tmp_path / "sparse_clean"
    output.mkdir()
    (output / "CLEAN_STOP_REQUEST.json").write_text(json.dumps({"request_id": "sparse-stop"}))
    engine = TrainingEngine(_sparse_config(), device="cpu")
    model = nn.Linear(1, 1)
    receipt = engine.fit(model, _ToyProvider(), output, identity={"run": "sparse-clean"},
                         arm="adaptive_detail", stop_after=10)
    assert receipt["completed_epoch"] == 1 and receipt["status"] == "clean_stopped"
    assert not list(output.glob("epoch_*_model.pt"))
    assert (output / "latest_model.pt").exists()
    resumed = engine.fit(model, _ToyProvider(), output, identity={"run": "sparse-clean"},
                         arm="adaptive_detail", stop_after=2, resume_checkpoint=output / "latest_model.pt")
    assert resumed["completed_epoch"] == 2


def test_unused_output_controls_preserve_legacy_checkpoint_identity():
    payload = runtime._engine_config_payload(_config())
    assert not {"monitor_epochs", "checkpoint_epochs", "latest_every", "curve_every"} & payload.keys()


@pytest.mark.parametrize("failed_sidecar", ["epoch_0010_model.pt", "best_by_field_mse_model.pt"])
def test_failed_terminal_sidecar_does_not_advance_latest_past_recoverable_age(tmp_path, monkeypatch, failed_sidecar):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args: None)
    original_save = runtime._atomic_torch_save

    def fail_terminal(path, payload):
        if path.name == failed_sidecar and payload["epoch"] == 10:
            raise OSError("injected terminal sidecar failure")
        original_save(path, payload)

    monkeypatch.setattr(runtime, "_atomic_torch_save", fail_terminal)
    engine = TrainingEngine(_sparse_config(), device="cpu")
    output = tmp_path / "interrupted_terminal"
    model = nn.Linear(1, 1)
    with pytest.raises(OSError, match="injected terminal"):
        engine.fit(model, _ToyProvider(), output, identity={"run": "recover"},
                   arm="adaptive_detail", stop_after=10)
    saved = torch.load(output / "latest_model.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 8
    monkeypatch.setattr(runtime, "_atomic_torch_save", original_save)
    result = engine.fit(model, _ToyProvider(), output, identity={"run": "recover"},
                        arm="adaptive_detail", stop_after=10, resume_checkpoint=output / "latest_model.pt")
    assert result["completed_epoch"] == 10
    for name in ("latest_model.pt", "epoch_0010_model.pt", "best_by_field_mse_model.pt"):
        assert torch.load(output / name, map_location="cpu", weights_only=False)["epoch"] == 10


def test_schedule_hits_declared_warmup_endpoints_and_cosine_final():
    schedule = ScheduleSpec(peak_lr=5.0e-5, warmup_start_lr=3.0e-6, warmup_epochs=20,
                           hold_through_epoch=1000, total_epochs=2500, final_lr=3.0e-6)
    assert schedule.value(1) == pytest.approx(3.0e-6)
    assert schedule.value(20) == pytest.approx(5.0e-5)
    assert schedule.value(1000) == pytest.approx(5.0e-5)
    assert schedule.value(2500) == pytest.approx(3.0e-6)


def _fit_warmup(tmp_path: Path):
    model = nn.Linear(1, 1)
    engine = TrainingEngine(_config(), device="cpu", selection=SelectionPolicy())
    engine.fit(model, _ToyProvider(), tmp_path / "warmup", identity={"run": "test"}, arm="warmup", stop_after=2)
    return model, tmp_path / "warmup" / "latest_model.pt"


def test_engine_uses_actual_final_macro_denominator_and_records_visits(tmp_path):
    model = nn.Linear(1, 1)
    provider = _ToyProvider()
    engine = TrainingEngine(_config(), device="cpu", selection=SelectionPolicy())
    receipt = engine.fit(model, provider, tmp_path / "warmup", identity={"run": "test"}, arm="warmup",
                         stop_after=2)
    history = torch.load(tmp_path / "warmup" / "latest_model.pt", map_location="cpu", weights_only=False)["history"]
    assert receipt["completed_epoch"] == 2
    assert history[0]["case_visits"] == 5
    assert history[0]["optimizer_updates"] == 2
    assert history[0]["microbatches"] == 3
    assert history[0]["partial_update_cases"] == 1
    assert history[0]["work_counts"]["query_rows"] == 5
    assert (tmp_path / "warmup" / "epoch_0001_model.pt").is_file()
    assert not (tmp_path / "warmup" / "best_by_response_guarded_model.pt").exists()
    summary = json.loads((tmp_path / "warmup" / "fit_summary.json").read_text())
    assert summary["best_response_guarded_score"] is None


def test_guarded_selector_is_a_verified_pointer_to_a_monitoring_checkpoint(tmp_path):
    model = nn.Linear(1, 1)
    engine = TrainingEngine(
        _config(), device="cpu",
        selection=SelectionPolicy(field_metric="field_score",
                                  response_guard_metric="response_guard_max_ratio",
                                  maximum_response_ratio=1.1),
    )
    output = tmp_path / "guarded"
    receipt = engine.fit(model, _GuardedToyProvider(), output, identity={"run": "guarded"},
                         arm="warmup", stop_after=2)
    payload = torch.load(output / "latest_model.pt", map_location="cpu", weights_only=False)
    rows = payload["history"]
    expected = min(
        (row for row in rows if row["validation"]["response_guard_max_ratio"] <= 1.1),
        key=lambda row: row["validation"]["field_score"],
    )
    pointer_path = output / "best_by_response_guarded_selection.json"
    pointer = json.loads(pointer_path.read_text())
    selected = output / pointer["checkpoint"]
    assert receipt["best_response_guarded_score"] == expected["validation"]["field_score"]
    assert pointer["selector"] == "response_guarded_field_score"
    assert pointer["epoch"] == expected["epoch"]
    assert pointer["field_score"] == expected["validation"]["field_score"]
    assert pointer["guard_value"] == expected["validation"]["response_guard_max_ratio"]
    assert selected.is_file()
    assert hashlib.sha256(selected.read_bytes()).hexdigest() == pointer["checkpoint_sha256"]
    assert not (output / "best_by_response_guarded_model.pt").exists()


def test_engine_rejects_a_denominator_that_does_not_match_the_target_prepass(tmp_path):
    model = nn.Linear(1, 1)
    engine = TrainingEngine(_config(), device="cpu")
    with pytest.raises(ValueError, match="differs from prepass"):
        engine.fit(model, _ToyProvider(bad_prepass=True), tmp_path / "bad", identity={"run": "bad"},
                   arm="warmup", stop_after=1)
    failed = json.loads((tmp_path / "bad" / "active_process.json").read_text())
    assert failed["status"] == "failed"
    assert failed["completed_epoch"] == 0
    assert failed["exception_type"] == "ValueError"
    assert failed["pid"] > 0
    assert failed["process_start_ticks"]
    assert not (tmp_path / "bad" / "latest_model.pt").exists()


def test_new_fit_cannot_overwrite_existing_history(tmp_path):
    _, latest = _fit_warmup(tmp_path)
    output = latest.parent
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    engine = TrainingEngine(_config(), device="cpu")
    with pytest.raises(ValueError, match="fresh run directory"):
        engine.fit(nn.Linear(1, 1), _ToyProvider(), output, identity={"run": "test"},
                   arm="warmup", stop_after=2)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}


def test_exact_resume_cannot_rewind_to_an_old_monitoring_checkpoint(tmp_path):
    _, latest = _fit_warmup(tmp_path)
    output = latest.parent
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    engine = TrainingEngine(_config(), device="cpu")
    with pytest.raises(ValueError, match="current latest_model.pt"):
        engine.fit(nn.Linear(1, 1), _ToyProvider(), output, identity={"run": "test"},
                   arm="warmup", stop_after=3, resume_checkpoint=output / "epoch_0001_model.pt")
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}


def test_another_process_lock_prevents_concurrent_checkpoint_writers(tmp_path):
    output = tmp_path / "locked"
    output.mkdir()
    engine = TrainingEngine(_config(), device="cpu")
    with (output / ".training.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another training process"):
            engine.fit(nn.Linear(1, 1), _ToyProvider(), output, identity={"run": "locked"},
                       arm="warmup", stop_after=1)
    assert not (output / "active_process.json").exists()
    assert not (output / "latest_model.pt").exists()
    # Releasing the owner's kernel lock permits the same new output to start.
    assert engine.fit(nn.Linear(1, 1), _ToyProvider(), output, identity={"run": "locked"},
                      arm="warmup", stop_after=1)["completed_epoch"] == 1


def test_disposable_preflight_uses_the_engine_update_path_without_checkpointing():
    model = nn.Linear(1, 1)
    provider = _ToyProvider()
    engine = TrainingEngine(_config(), device="cpu")
    receipt = engine.preflight_one_update(model, provider, optimizer_seed=None)
    assert receipt["status"] == "disposable_cpu_preflight_update"
    assert receipt["case_keys"] == [0]
    assert receipt["optimizer_updates"] == 1
    assert receipt["query_sampling_sha256"] is not None
    assert receipt["denominators"] == {"native": 1.0}
    assert len(provider.training_mode_seen) == 1
    assert all(provider.training_mode_seen)


def test_clean_stop_is_consumed_before_exact_resume(tmp_path):
    model = nn.Linear(1, 1)
    provider = _ToyProvider()
    engine = TrainingEngine(_config(), device="cpu")
    output = tmp_path / "clean_stop"
    output.mkdir()
    request = {"request_id": "req-clean-stop-001", "requested_unix": 1.0, "requested_by": "unit-test"}
    (output / "CLEAN_STOP_REQUEST.json").write_text(json.dumps(request) + "\n", encoding="utf-8")
    stopped = engine.fit(model, provider, output, identity={"run": "clean-stop"}, arm="warmup", stop_after=4)
    assert stopped["status"] == "clean_stopped"
    assert stopped["completed_epoch"] == 1
    assert not (output / "CLEAN_STOP_REQUEST.json").exists()
    acknowledgement = json.loads((output / "clean_stop_acknowledged.json").read_text())
    assert acknowledgement["request_id"] == request["request_id"]
    assert acknowledgement["epoch"] == 1
    assert acknowledgement["phase"] == "warmup"

    # Simulate a crash between writing the acknowledgement and deleting the
    # marker. Resume may consume only this matching acknowledged request.
    (output / "CLEAN_STOP_REQUEST.json").write_text(json.dumps(request) + "\n", encoding="utf-8")

    resumed = engine.fit(model, provider, output, identity={"run": "clean-stop"}, arm="warmup", stop_after=2,
                         resume_checkpoint=output / "latest_model.pt")
    assert resumed["completed_epoch"] == 2
    assert resumed["status"] == "completed"
    assert not (output / "CLEAN_STOP_REQUEST.json").exists()
    consumed = list(output.glob("clean_stop_consumed_*.json"))
    assert len(consumed) == 1
    consumed_payload = json.loads(consumed[0].read_text())
    assert consumed_payload["request"] == request
    assert consumed_payload["acknowledgement"] == acknowledgement
    assert consumed_payload["consumed_on_resume"] is True


def test_shared_warmup_branches_and_exact_resume_reproduce_same_endpoint(tmp_path):
    torch.manual_seed(99)
    warm_model, warm_checkpoint = _fit_warmup(tmp_path)
    warm_state = copy.deepcopy(warm_model.state_dict())
    engine = TrainingEngine(_config(), device="cpu", selection=SelectionPolicy())
    identity = {"run": "test"}

    interrupted = nn.Linear(1, 1)
    first = engine.fit(interrupted, _ToyProvider(), tmp_path / "adaptive", identity=identity,
                       arm="adaptive_detail", stop_after=3, branch_from_checkpoint=warm_checkpoint)
    assert first["completed_epoch"] == 3
    resumed = nn.Linear(1, 1)
    engine.fit(resumed, _ToyProvider(), tmp_path / "adaptive", identity=identity,
               arm="adaptive_detail", stop_after=4,
               resume_checkpoint=tmp_path / "adaptive" / "latest_model.pt")

    uninterrupted = nn.Linear(1, 1)
    engine.fit(uninterrupted, _ToyProvider(), tmp_path / "full", identity=identity,
               arm="full_detail", stop_after=4, branch_from_checkpoint=warm_checkpoint)
    for key, value in uninterrupted.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
    assert all(torch.equal(warm_state[key], warm_model.state_dict()[key]) for key in warm_state)
    resumed_history = torch.load(tmp_path / "adaptive" / "latest_model.pt", map_location="cpu",
                                 weights_only=False)["history"]
    full_history = torch.load(tmp_path / "full" / "latest_model.pt", map_location="cpu",
                              weights_only=False)["history"]
    assert [row["query_sampling_sha256"] for row in resumed_history] == [
        row["query_sampling_sha256"] for row in full_history
    ]


def test_sampling_key_uses_matched_arm_independent_query_seeds():
    full = SamplingKey(17, 601, 2, 1, "soft", "full_detail")
    adaptive = SamplingKey(17, 601, 2, 1, "soft", "adaptive_detail")
    assert full.seed_for("case-033", "fluid") == adaptive.seed_for("case-033", "fluid")
    assert full.seed_for("case-033", "fluid") != full.seed_for("case-034", "fluid")
    assert full.numpy_rng("receiver").integers(2**31) == adaptive.numpy_rng("receiver").integers(2**31)


def test_engine_temperature_anneals_on_absolute_soft_epochs():
    config = EngineConfig(seed=1, microbatch_cases=2, effective_cases=4, total_epochs=2500,
                         warmup_epochs=500, open_through_epoch=600, soft_through_epoch=800)
    assert config.temperature_for_epoch(600) == pytest.approx(1.0)
    assert config.temperature_for_epoch(601) == pytest.approx(1.0)
    assert config.temperature_for_epoch(800) == pytest.approx(0.1)
    assert config.temperature_for_epoch(801) == pytest.approx(0.1)


def test_validation_always_uses_hard_route_during_open_training(tmp_path):
    model = nn.Linear(1, 1)
    provider = _ToyProvider()
    engine = TrainingEngine(_config(), device="cpu", selection=SelectionPolicy())
    engine.fit(model, provider, tmp_path / "warmup", identity={"run": "test"}, arm="warmup", stop_after=1)
    assert provider.validation_phases
    assert all(phase == "hard" for phase, _mode, _epoch, _temperature in provider.validation_phases)
    saved = torch.load(tmp_path / "warmup" / "latest_model.pt", map_location="cpu", weights_only=False)
    assert saved["history"][0]["validation"]["validation_route_phase"] == "hard"
    assert provider.training_mode_seen
    assert all(provider.training_mode_seen)


def test_branch_checks_all_sealed_identity_fields(tmp_path):
    _, warm_checkpoint = _fit_warmup(tmp_path)
    model = nn.Linear(1, 1)
    engine = TrainingEngine(_config(), device="cpu")
    with pytest.raises(ValueError, match="identity, engine config"):
        engine.fit(model, _ToyProvider(), tmp_path / "bad_identity", identity={"run": "changed"},
                   arm="adaptive_detail", stop_after=3, branch_from_checkpoint=warm_checkpoint)


def test_branch_requires_literal_warmup_boundary(tmp_path):
    _, warm_checkpoint = _fit_warmup(tmp_path)
    model = nn.Linear(1, 1)
    engine = TrainingEngine(_config(), device="cpu")
    payload = torch.load(warm_checkpoint, map_location="cpu", weights_only=False)
    payload["epoch"] = 1
    bad = tmp_path / "not_warmup_boundary.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="common warmup boundary"):
        engine.fit(model, _ToyProvider(), tmp_path / "bad_branch", identity={"run": "test"},
                   arm="adaptive_detail", stop_after=3, branch_from_checkpoint=bad)


def test_microbatch_amendment_preserves_weights_moments_history_and_future_exact_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args: None)
    output = tmp_path / "continuation"
    model = nn.Linear(1, 1)
    config = _sparse_config()
    identity = {"run": "preserved"}
    TrainingEngine(config, device="cpu").fit(model, _ToyProvider(), output,
        identity=identity, arm="adaptive_detail", stop_after=4)
    source = torch.load(output / "latest_model.pt", map_location="cpu", weights_only=False)
    new_engine = TrainingEngine(replace(config, microbatch_cases=4), device="cpu")
    initial_steps = []
    original_optimizer = new_engine._make_optimizer

    def inspect_optimizer(*args, **kwargs):
        optimizer, specs = original_optimizer(*args, **kwargs)
        for name, value in source["model_state_dict"].items():
            assert torch.equal(model.state_dict()[name], value)
        for name, parameter in model.named_parameters():
            for key, value in source["optimizer_state_by_name"]["state_by_name"][name].items():
                if torch.is_tensor(value):
                    assert torch.equal(optimizer.state[parameter][key], value)
        initial_steps.extend(int(state["step"].item()) for state in optimizer.state.values())
        return optimizer, specs

    monkeypatch.setattr(new_engine, "_make_optimizer", inspect_optimizer)
    with pytest.raises(ValueError, match="identity, engine config"):
        new_engine.fit(model, _ToyProvider(), output, identity=identity,
            arm="adaptive_detail", stop_after=6, resume_checkpoint=output / "latest_model.pt")
    new_engine.fit(model, _ToyProvider(), output, identity=identity, arm="adaptive_detail", stop_after=6,
        resume_checkpoint=output / "latest_model.pt", allow_microbatch_change=True)
    assert initial_steps == [8, 8]
    amended = torch.load(output / "latest_model.pt", map_location="cpu", weights_only=False)
    assert amended["history"][:4] == source["history"]
    assert amended["history"][4]["microbatches"] == 2
    record = amended["resume_amendments"][0]
    assert record["source_epoch"] == 4 and record["next_epoch"] == 5
    assert record["preserved_case_visits"] == 20 and record["preserved_optimizer_updates"] == 8
    assert record["source_microbatch_cases"] == 2 and record["microbatch_cases"] == 4
    assert json.loads((output / "microbatch_amendment_epoch_0005.json").read_text()) == record
    for state in amended["optimizer_state_by_name"]["state_by_name"].values():
        assert int(state["step"].item()) == 12
    # The next continuation is exact under the amended identity; no further exception is needed.
    engine = TrainingEngine(replace(config, microbatch_cases=4), device="cpu")
    engine.fit(model, _ToyProvider(), output, identity=identity, arm="adaptive_detail", stop_after=10,
        resume_checkpoint=output / "latest_model.pt")
    final = torch.load(output / "latest_model.pt", map_location="cpu", weights_only=False)
    assert final["resume_amendments"] == amended["resume_amendments"]
    assert sum(row["case_visits"] for row in final["history"]) == 50
    assert sum(row["optimizer_updates"] for row in final["history"]) == 20


@pytest.mark.parametrize("changed", ["effective_cases", "seed", "provider_identity", "selection_policy",
                                     "optimizer_schedule_contract", "recipe_sha256"])
def test_explicit_microbatch_exception_rejects_any_other_identity_change(changed):
    saved = {"engine_config": {"microbatch_cases": 2, "effective_cases": 4, "seed": 17},
             "provider_identity": {"dataset": "fixed"}, "selection_policy": {"field": "native"},
             "optimizer_schedule_contract": ["sealed"], "recipe_sha256": "sealed"}
    current = copy.deepcopy(saved)
    current["engine_config"]["microbatch_cases"] = 4
    if changed in ("effective_cases", "seed"):
        current["engine_config"][changed] += 1
    else:
        current[changed] = "changed"
    original = copy.deepcopy(saved)
    with pytest.raises(ValueError, match="identity, engine config"):
        runtime._resume_identity_amendment(saved, current, allow_microbatch_change=True)
    assert saved == original
