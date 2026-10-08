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
from honf_runtime.run_layout import RunLayout, resolve_checkpoint
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


def _artifact(run_dir: Path, name: str) -> Path:
    return RunLayout(run_dir).read_path(name)


def _checkpoint(run_dir: Path, selector: str = "latest") -> Path:
    return resolve_checkpoint(run_dir, selector)


def _tree_hash(run_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_dir.rglob("*")
        if path.is_file()
    }


def test_sparse_outputs_keep_only_milestones_and_update_real_curves(tmp_path):
    provider = _ToyProvider()
    output = tmp_path / "sparse"
    engine = TrainingEngine(_sparse_config(), device="cpu")
    engine.fit(nn.Linear(1, 1), provider, output, identity={"run": "sparse"},
               arm="adaptive_detail", stop_after=10)
    history = json.loads(_artifact(output, "history.json").read_text())
    assert [row["epoch"] for row in history if "validation" in row] == [2, 4, 8, 10]
    assert {path.name for path in (output / "checkpoints").glob("epoch_*_model.pt")} == {
        "epoch_0002_model.pt", "epoch_0006_model.pt", "epoch_0010_model.pt"}
    for name in ("loss_curves.pdf", "loss_curves.png"):
        assert _artifact(output, name).stat().st_size > 0
    assert _checkpoint(output, "best_field").stat().st_size > 0
    progress = json.loads(_artifact(output, "progress.json").read_text())
    assert progress["completed_epoch"] == progress["latest_checkpoint_epoch"] == 10
    assert progress["completed_case_visits"] == 50
    assert progress["completed_optimizer_updates"] == 20
    assert not (output / "last.pt").exists()
    assert not (output / "checkpoints" / "last.pt").exists()
    assert torch.load(_checkpoint(output), map_location="cpu", weights_only=False)["epoch"] == 10
    summary = json.loads(_artifact(output, "fit_summary.json").read_text())
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
               arm="adaptive_detail", stop_after=10, resume_checkpoint=_checkpoint(output))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    actual_numpy_rng = np.random.get_state()
    assert np.array_equal(actual_numpy_rng[1], expected_numpy_rng[1])
    assert actual_numpy_rng[2:] == expected_numpy_rng[2:]
    for name, value in uninterrupted.state_dict().items():
        assert torch.equal(value, interrupted.state_dict()[name])
    summary = json.loads(_artifact(output, "fit_summary.json").read_text())
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
    assert not list((output / "checkpoints").glob("epoch_*_model.pt"))
    assert _checkpoint(output).exists()
    resumed = engine.fit(model, _ToyProvider(), output, identity={"run": "sparse-clean"},
                         arm="adaptive_detail", stop_after=2, resume_checkpoint=_checkpoint(output))
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
    saved = torch.load(_checkpoint(output), map_location="cpu", weights_only=False)
    assert saved["epoch"] == 8
    monkeypatch.setattr(runtime, "_atomic_torch_save", original_save)
    result = engine.fit(model, _ToyProvider(), output, identity={"run": "recover"},
                        arm="adaptive_detail", stop_after=10, resume_checkpoint=_checkpoint(output))
    assert result["completed_epoch"] == 10
    for selector in ("latest", "epoch_0010_model.pt", "best_field"):
        assert torch.load(_checkpoint(output, selector), map_location="cpu", weights_only=False)["epoch"] == 10


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
    return model, _checkpoint(tmp_path / "warmup")


def test_engine_uses_actual_final_macro_denominator_and_records_visits(tmp_path):
    model = nn.Linear(1, 1)
    provider = _ToyProvider()
    engine = TrainingEngine(_config(), device="cpu", selection=SelectionPolicy())
    receipt = engine.fit(model, provider, tmp_path / "warmup", identity={"run": "test"}, arm="warmup",
                         stop_after=2)
    history = torch.load(_checkpoint(tmp_path / "warmup"), map_location="cpu", weights_only=False)["history"]
    assert receipt["completed_epoch"] == 2
    assert history[0]["case_visits"] == 5
    assert history[0]["optimizer_updates"] == 2
    assert history[0]["microbatches"] == 3
    assert history[0]["partial_update_cases"] == 1
    assert history[0]["work_counts"]["query_rows"] == 5
    assert _checkpoint(tmp_path / "warmup", "epoch_0001_model.pt").is_file()
    assert not (tmp_path / "warmup" / "best_by_response_guarded_model.pt").exists()
    summary = json.loads(_artifact(tmp_path / "warmup", "fit_summary.json").read_text())
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
    payload = torch.load(_checkpoint(output), map_location="cpu", weights_only=False)
    rows = payload["history"]
    expected = min(
        (row for row in rows if row["validation"]["response_guard_max_ratio"] <= 1.1),
        key=lambda row: row["validation"]["field_score"],
    )
    pointer_path = _artifact(output, "best_by_response_guarded_selection.json")
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
    failed = json.loads(_artifact(tmp_path / "bad", "active_process.json").read_text())
    assert failed["status"] == "failed"
    assert failed["completed_epoch"] == 0
    assert failed["exception_type"] == "ValueError"
    assert failed["pid"] > 0
    assert failed["process_start_ticks"]
    assert not _checkpoint(tmp_path / "bad").exists()


def test_new_fit_cannot_overwrite_existing_history(tmp_path):
    _, latest = _fit_warmup(tmp_path)
    output = latest.parent.parent
    before = _tree_hash(output)
    engine = TrainingEngine(_config(), device="cpu")
    with pytest.raises(ValueError, match="fresh run directory"):
        engine.fit(nn.Linear(1, 1), _ToyProvider(), output, identity={"run": "test"},
                   arm="warmup", stop_after=2)
    assert before == _tree_hash(output)


def test_exact_resume_cannot_rewind_to_an_old_monitoring_checkpoint(tmp_path):
    _, latest = _fit_warmup(tmp_path)
    output = latest.parent.parent
    before = _tree_hash(output)
    engine = TrainingEngine(_config(), device="cpu")
    with pytest.raises(ValueError, match="current latest checkpoint"):
        engine.fit(nn.Linear(1, 1), _ToyProvider(), output, identity={"run": "test"},
                   arm="warmup", stop_after=3, resume_checkpoint=_checkpoint(output, "epoch_0001_model.pt"))
    assert before == _tree_hash(output)


def test_legacy_exact_resume_uses_root_latest_over_stale_checkpoint_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args: None)
    output = tmp_path / "legacy_resume"
    engine = TrainingEngine(_config(), device="cpu")
    model = nn.Linear(1, 1)
    engine.fit(model, _ToyProvider(), output, identity={"run": "legacy"}, arm="warmup", stop_after=2)

    latest = _checkpoint(output)
    latest_payload = torch.load(latest, map_location="cpu", weights_only=False)
    torch.load(_checkpoint(output, "epoch_0001_model.pt"), map_location="cpu", weights_only=False)
    (output / "artifact_layout.json").unlink()
    (output / "latest_model.pt").write_bytes(latest.read_bytes())
    (output / "checkpoints" / "latest.pt").write_bytes(b"stale alias")

    assert resolve_checkpoint(output, "latest") == output / "latest_model.pt"
    resumed = engine.fit(
        model,
        _ToyProvider(),
        output,
        identity={"run": "legacy"},
        arm="warmup",
        stop_after=3,
        resume_checkpoint=output / "latest_model.pt",
    )

    assert latest_payload["epoch"] == 2
    assert resumed["completed_epoch"] == 3
    assert torch.load(output / "latest_model.pt", map_location="cpu", weights_only=False)["epoch"] == 3
    assert (output / "checkpoints" / "latest.pt").read_bytes() == b"stale alias"


def test_another_process_lock_prevents_concurrent_checkpoint_writers(tmp_path):
    output = tmp_path / "locked"
    output.mkdir()
    engine = TrainingEngine(_config(), device="cpu")
    with (output / ".training.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another training process"):
            engine.fit(nn.Linear(1, 1), _ToyProvider(), output, identity={"run": "locked"},
                       arm="warmup", stop_after=1)
    assert not _artifact(output, "active_process.json").exists()
    assert not _checkpoint(output).exists()
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
    acknowledgement = json.loads(_artifact(output, "clean_stop_acknowledged.json").read_text())
    assert acknowledgement["request_id"] == request["request_id"]
    assert acknowledgement["epoch"] == 1
    assert acknowledgement["phase"] == "warmup"

    # Simulate a crash between writing the acknowledgement and deleting the
    # marker. Resume may consume only this matching acknowledged request.
    (output / "CLEAN_STOP_REQUEST.json").write_text(json.dumps(request) + "\n", encoding="utf-8")

    resumed = engine.fit(model, provider, output, identity={"run": "clean-stop"}, arm="warmup", stop_after=2,
                         resume_checkpoint=_checkpoint(output))
    assert resumed["completed_epoch"] == 2
    assert resumed["status"] == "completed"
    assert not (output / "CLEAN_STOP_REQUEST.json").exists()
    consumed = list((output / "logs").glob("clean_stop_consumed_*.json"))
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
               resume_checkpoint=_checkpoint(tmp_path / "adaptive"))

    uninterrupted = nn.Linear(1, 1)
    engine.fit(uninterrupted, _ToyProvider(), tmp_path / "full", identity=identity,
               arm="full_detail", stop_after=4, branch_from_checkpoint=warm_checkpoint)
    for key, value in uninterrupted.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
    assert all(torch.equal(warm_state[key], warm_model.state_dict()[key]) for key in warm_state)
    resumed_history = torch.load(_checkpoint(tmp_path / "adaptive"), map_location="cpu",
                                 weights_only=False)["history"]
    full_history = torch.load(_checkpoint(tmp_path / "full"), map_location="cpu",
                              weights_only=False)["history"]
    assert [row["query_sampling_sha256"] for row in resumed_history] == [
        row["query_sampling_sha256"] for row in full_history
    ]


def _compact_declaration(tmp_path):
    evidence = tmp_path / "train_seams.json"
    evidence.write_text(json.dumps({"partition": "two input-selected TRAIN scenes",
        "TEST_read": False, "optimizer_updates": 0, "solver_attempts": 0,
        "rows": [{"consequential": True}]}), encoding="utf-8")
    return {"kind": "compact_c1_gate_branch", "from_gate_version": "hard_v1",
        "to_gate_version": "compact_c1_v1", "gate_transition": [0.35, 0.65],
        "evidence_path": str(evidence),
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "reason": "Input-only TRAIN seam exceeds the declared physical threshold."}


def _wind_gate_identities():
    parent = {"run_id": "matched", "wind_recipe_id": "balanced_v1",
        "wind_recipe_sha256": "parent-recipe", "sampling_dataset_id": "wind:frozen",
        "provider_identity": {"manifest_sha256": "fixed24", "normalizer": "train-only",
            "resolved_recipe_sha256": "parent-recipe",
            "resolved_recipe": {"recipe_id": "balanced_v1", "recipe_sha256": "parent-recipe",
                "query_count": 1024, "hidden_dim": 64, "native_objective": "component_balanced_v1"}},
        "engine_config": {"effective_cases": 24, "microbatch_cases": 24,
            "sampling_version": "case_epoch_v1", "total_epochs": 2500},
        "optimizer_schedule_contract": [{"peak_lr": 5e-5, "hold_through_epoch": 1000}]}
    child = copy.deepcopy(parent)
    child["wind_recipe_id"] = "balanced_c1_v1"
    child["wind_recipe_sha256"] = "child-recipe"
    child["provider_identity"]["resolved_recipe_sha256"] = "child-recipe"
    child["provider_identity"]["resolved_recipe"].update(recipe_id="balanced_c1_v1",
        recipe_sha256="child-recipe", gate_version="compact_c1_v1", gate_transition=[0.35, 0.65])
    return parent, child


def test_compact_branch_preserves_every_other_wind_binding(tmp_path):
    parent, child = _wind_gate_identities()
    before = copy.deepcopy(child)
    receipt = runtime._branch_mathematical_route_amendment(parent, child, _compact_declaration(tmp_path))
    assert receipt["to_gate_version"] == "compact_c1_v1"
    assert child == before


@pytest.mark.parametrize("binding", ["manifest", "normalizer", "objective", "queries", "width",
    "effective_batch", "microbatch", "sampler", "schedule", "run"])
def test_compact_branch_rejects_other_scientific_changes(tmp_path, binding):
    parent, child = _wind_gate_identities()
    provider = child["provider_identity"]
    recipe = provider["resolved_recipe"]
    if binding == "manifest": provider["manifest_sha256"] = "different"
    elif binding == "normalizer": provider["normalizer"] = "different"
    elif binding == "objective": recipe["native_objective"] = "different"
    elif binding == "queries": recipe["query_count"] = 4096
    elif binding == "width": recipe["hidden_dim"] = 128
    elif binding == "effective_batch": child["engine_config"]["effective_cases"] = 48
    elif binding == "microbatch": child["engine_config"]["microbatch_cases"] = 6
    elif binding == "sampler": child["engine_config"]["sampling_version"] = "legacy_v1"
    elif binding == "schedule": child["optimizer_schedule_contract"][0]["peak_lr"] = 1e-4
    else: child["run_id"] = "different"
    with pytest.raises(ValueError, match="any other experiment binding"):
        runtime._branch_mathematical_route_amendment(parent, child, _compact_declaration(tmp_path))


@pytest.mark.parametrize("bad_evidence", ["hash", "test", "updates", "solves", "partition", "consequential"])
def test_compact_branch_requires_unmodified_input_only_train_diagnosis(tmp_path, bad_evidence):
    parent, child = _wind_gate_identities()
    declaration = _compact_declaration(tmp_path)
    evidence = Path(declaration["evidence_path"])
    payload = json.loads(evidence.read_text())
    if bad_evidence == "test": payload["TEST_read"] = True
    elif bad_evidence == "updates": payload["optimizer_updates"] = 1
    elif bad_evidence == "solves": payload["solver_attempts"] = 1
    elif bad_evidence == "partition": payload["partition"] = "DEV"
    elif bad_evidence == "consequential": payload["rows"][0]["consequential"] = False
    evidence.write_text(json.dumps(payload) + " ", encoding="utf-8")
    if bad_evidence != "hash":
        declaration["evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="diagnosis|TRAIN seam"):
        runtime._branch_mathematical_route_amendment(parent, child, declaration)


def test_compact_child_records_amendment_and_keeps_exact_resume_strict(tmp_path):
    class GateProvider(_ToyProvider):
        def __init__(self, gate="hard_v1"):
            super().__init__()
            self.gate = gate

        def identity_payload(self):
            payload = super().identity_payload()
            if self.gate != "hard_v1":
                payload.update(gate_version=self.gate, gate_transition=[0.35, 0.65])
            return payload

    engine = TrainingEngine(_config(), device="cpu", selection=SelectionPolicy())
    engine.fit(nn.Linear(1, 1), GateProvider(), tmp_path / "parent", identity={"run": "gate"},
        arm="warmup", stop_after=2)
    parent = _checkpoint(tmp_path / "parent")
    child_dir = tmp_path / "child"
    engine.fit(nn.Linear(1, 1), GateProvider("compact_c1_v1"), child_dir,
        identity={"run": "gate"}, arm="adaptive_detail", stop_after=3,
        branch_from_checkpoint=parent, mathematical_route_branch=_compact_declaration(tmp_path))
    payload = torch.load(_checkpoint(child_dir), map_location="cpu", weights_only=False)
    assert payload["branch_parent"]["sha256"] == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert payload["resume_amendments"][-1]["source_epoch"] == 2
    assert payload["experiment_identity"]["provider_identity"]["gate_version"] == "compact_c1_v1"
    assert _artifact(child_dir, "mathematical_route_amendment_epoch_0003.json").is_file()
    with pytest.raises(ValueError, match="different|differs|identity|Identity"):
        engine.fit(nn.Linear(1, 1), GateProvider(), child_dir, identity={"run": "gate"},
            arm="adaptive_detail", stop_after=4, resume_checkpoint=_checkpoint(child_dir))
    engine.fit(nn.Linear(1, 1), GateProvider("compact_c1_v1"), child_dir,
        identity={"run": "gate"}, arm="adaptive_detail", stop_after=4,
        resume_checkpoint=_checkpoint(child_dir))
    resumed = torch.load(_checkpoint(child_dir), map_location="cpu", weights_only=False)
    assert resumed["branch_parent"] == payload["branch_parent"]


def test_sampling_key_uses_matched_arm_independent_query_seeds():
    full = SamplingKey(17, 601, 2, 1, "soft", "full_detail")
    adaptive = SamplingKey(17, 601, 2, 1, "soft", "adaptive_detail")
    assert full.seed_for("case-033", "fluid") == adaptive.seed_for("case-033", "fluid")
    assert full.seed_for("case-033", "fluid") != full.seed_for("case-034", "fluid")
    assert full.numpy_rng("receiver").integers(2**31) == adaptive.numpy_rng("receiver").integers(2**31)


def test_case_epoch_stream_is_packing_independent_and_prefix_consistent():
    full = SamplingKey(
        17, 601, 2, 1, "soft", "full_detail",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id="toy:fixed-membership",
    )
    repacked = SamplingKey(
        17, 601, 99, 7, "hard", "adaptive_detail",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION, dataset_id="toy:fixed-membership",
    )
    short = full.numpy_rng("case-033", "fluid", "primary").random(31)
    long_rng = repacked.numpy_rng("case-033", "fluid", "primary")
    long = long_rng.random(97)
    np.testing.assert_array_equal(short, long[:len(short)])
    offset = repacked.numpy_rng("case-033", "fluid", "primary", draw_start=31).random(66)
    np.testing.assert_array_equal(offset, long[31:])

    short_ids = full.native_indices(1009, 41, "case-033", "fluid", "primary")
    long_ids = repacked.native_indices(1009, 91, "case-033", "fluid", "primary")
    np.testing.assert_array_equal(short_ids, long_ids[:len(short_ids)])
    assert len(np.unique(long_ids)) == len(long_ids)
    for changed in (
        replace(full, epoch=602),
        replace(full, dataset_id="toy:other-membership"),
        replace(full, seed=18),
    ):
        assert not np.array_equal(long, changed.numpy_rng("case-033", "fluid", "primary").random(97))


class _PackingSamplerProvider(_ToyProvider):
    def make_batch(self, case_keys, key: SamplingKey):
        rows = []
        for case_id in case_keys:
            ids = key.native_indices(257, 9, f"case-{case_id}", "native", "primary")
            rows.append(ids)
        ids = np.stack(rows).astype(np.float32)
        x = torch.as_tensor(ids[..., None] / 257.0)
        case_offset = torch.as_tensor(case_keys, dtype=torch.float32)[:, None, None] / 100.0
        y = 0.75 * x + case_offset
        receipt = hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()
        return TaskBatch(scene_inputs=x, receivers=None, targets=y,
                         auxiliary={"query_sampling_sha256": receipt}, case_keys=tuple(case_keys))


def test_packing_changes_preserve_samples_denominators_gradients_and_fp32_update():
    provider_a = _PackingSamplerProvider()
    provider_b = _PackingSamplerProvider()
    version = SamplingKey.CASE_EPOCH_VERSION
    base_key = SamplingKey(17, 1, 0, 0, "warmup", "full_detail",
                           sampling_version=version, dataset_id="toy:fixed-membership")
    cases = (0, 1, 2, 3)

    def batches(provider, microbatch_cases):
        output = []
        sampled = {}
        for micro_index, start in enumerate(range(0, len(cases), microbatch_cases)):
            selected = cases[start:start + microbatch_cases]
            key = replace(base_key, microbatch_index=micro_index)
            batch = provider.make_batch(selected, key)
            native_ids = (batch.scene_inputs[..., 0] * 257.0).round().to(torch.int64).tolist()
            for case_id, ids in zip(selected, native_ids, strict=True):
                sampled[case_id] = ids
            output.append(batch)
        return tuple(output), sampled

    packed_a, samples_a = batches(provider_a, 2)
    packed_b, samples_b = batches(provider_b, 4)
    assert samples_a == samples_b

    engine = TrainingEngine(
        replace(_config(), microbatch_cases=2, effective_cases=4, sampling_version=version), device="cpu",
    )
    models = [nn.Linear(1, 1), nn.Linear(1, 1)]
    initial = {name: value.detach().clone() for name, value in models[0].state_dict().items()}
    for model in models:
        model.load_state_dict(initial)
    snapshots = []
    observed = []
    for model, provider, microbatches in zip(models, (provider_a, provider_b), (packed_a, packed_b), strict=True):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        step = optimizer.step

        def capture_step(model=model, step=step):
            snapshots.append({name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()})
            step()

        optimizer.step = capture_step
        losses, _work, denominators, _sampling_hash = engine._run_update(
            model, provider, optimizer, microbatches, phase="warmup", arm="full_detail",
            epoch=1, update_index=0,
        )
        observed.append((losses, denominators))
    assert observed[0][1] == observed[1][1] == {"native": 4 * 9}
    assert observed[0][0]["native"] == pytest.approx(observed[1][0]["native"], rel=1e-6, abs=1e-7)
    for name in snapshots[0]:
        torch.testing.assert_close(snapshots[0][name], snapshots[1][name], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(models[0].state_dict()[name], models[1].state_dict()[name], rtol=1e-6, atol=1e-7)


def test_case_epoch_sampling_recipe_is_stored_and_resume_is_strict(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args: None)
    config = replace(_sparse_config(), sampling_version=SamplingKey.CASE_EPOCH_VERSION)
    identity = {"run": "case-epoch-resume"}
    torch.manual_seed(820)
    initial_model = nn.Linear(1, 1)
    initial = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    uninterrupted = nn.Linear(1, 1)
    uninterrupted.load_state_dict(initial)
    torch.manual_seed(91)
    TrainingEngine(config, device="cpu").fit(
        uninterrupted, _PackingSamplerProvider(), tmp_path / "case_epoch_full", identity=identity,
        arm="adaptive_detail", stop_after=6,
    )
    expected_rng = torch.get_rng_state().clone()

    interrupted = nn.Linear(1, 1)
    interrupted.load_state_dict(initial)
    output = tmp_path / "case_epoch_resume"
    torch.manual_seed(91)
    TrainingEngine(config, device="cpu").fit(
        interrupted, _PackingSamplerProvider(), output, identity=identity,
        arm="adaptive_detail", stop_after=4,
    )
    TrainingEngine(config, device="cpu").fit(
        interrupted, _PackingSamplerProvider(), output, identity=identity,
        arm="adaptive_detail", stop_after=6, resume_checkpoint=_checkpoint(output),
    )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, interrupted.state_dict()[name], rtol=0.0, atol=0.0)
    saved = torch.load(_checkpoint(output), map_location="cpu", weights_only=False)
    assert saved["experiment_identity"]["engine_config"]["sampling_version"] == SamplingKey.CASE_EPOCH_VERSION

    legacy_engine = TrainingEngine(replace(config, sampling_version=SamplingKey.LEGACY_VERSION), device="cpu")
    with pytest.raises(ValueError, match="identity, engine config"):
        legacy_engine.fit(
            interrupted, _PackingSamplerProvider(), output, identity=identity,
            arm="adaptive_detail", stop_after=7, resume_checkpoint=_checkpoint(output),
        )


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
    saved = torch.load(_checkpoint(tmp_path / "warmup"), map_location="cpu", weights_only=False)
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
    source = torch.load(_checkpoint(output), map_location="cpu", weights_only=False)
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
            arm="adaptive_detail", stop_after=6, resume_checkpoint=_checkpoint(output))
    new_engine.fit(model, _ToyProvider(), output, identity=identity, arm="adaptive_detail", stop_after=6,
        resume_checkpoint=_checkpoint(output), allow_microbatch_change=True)
    assert initial_steps == [8, 8]
    amended = torch.load(_checkpoint(output), map_location="cpu", weights_only=False)
    assert amended["history"][:4] == source["history"]
    assert amended["history"][4]["microbatches"] == 2
    record = amended["resume_amendments"][0]
    assert record["source_epoch"] == 4 and record["next_epoch"] == 5
    assert record["preserved_case_visits"] == 20 and record["preserved_optimizer_updates"] == 8
    assert record["source_microbatch_cases"] == 2 and record["microbatch_cases"] == 4
    assert json.loads(_artifact(output, "microbatch_amendment_epoch_0005.json").read_text()) == record
    for state in amended["optimizer_state_by_name"]["state_by_name"].values():
        assert int(state["step"].item()) == 12
    # The next continuation is exact under the amended identity; no further exception is needed.
    engine = TrainingEngine(replace(config, microbatch_cases=4), device="cpu")
    engine.fit(model, _ToyProvider(), output, identity=identity, arm="adaptive_detail", stop_after=10,
        resume_checkpoint=_checkpoint(output))
    final = torch.load(_checkpoint(output), map_location="cpu", weights_only=False)
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
