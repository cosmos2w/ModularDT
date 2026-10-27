from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
for _directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.contextual_frontier_training import (
    _engine_config,
    _run_identity,
    load_contextual_frontier_config,
)
from honf_runtime.research_training import ResearchTrainingEngine
from honf_runtime.unified_training import (
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    SelectionPolicy,
    TaskBatch,
    _engine_config_payload,
    _optimizer_spec_payload,
    _resume_identity_amendment,
    _sampling_dataset_id,
)


CONFIGS = PROJECT / "configs" / "research"
CASES = (
    ("thermal_r13_6c_faithful_train600_formal_v1.json",
     "interaction_reused_faithful_member_routing_training_v1"),
    ("thermal_r13_6h_typed_organization_train600_formal_v1.json",
     "interaction_reused_typed_organization_member_routing_training_v1"),
)


class _TinyModel(torch.nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        self.architecture = arm
        self.linear = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(0.25)
            self.linear.bias.fill_(0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class _TinyProvider:
    def __init__(self, model: _TinyModel, config: dict):
        self.model = model
        self.optimizer = config["optimizer"]
        self.schedule_horizon = int(config["training"]["optimizer_schedule_horizon"])
        self.training_case_ids = ("train-a", "train-b")
        self.validation_case_ids = ("dev-a", "dev-b")
        self.calls = 0
        self.draws: list[float] = []
        self.sampler_epochs: list[int] = []

    def identity_payload(self):
        return {
            "schema": "r13_6_horizon_toy_provider_v1",
            "dataset": "thermal",
            "architecture": self.model.architecture,
            "training_case_ids": list(self.training_case_ids),
            "validation_case_ids": list(self.validation_case_ids),
        }

    def epoch_cases(self, epoch: int, seed: int):
        del seed
        self.sampler_epochs.append(int(epoch))
        return list(self.training_case_ids)

    @staticmethod
    def _batch(case_ids, shift: float = 0.0):
        x_by_case = {"train-a": 0.0, "train-b": 1.0, "dev-a": 0.5, "dev-b": 1.5}
        inputs = torch.tensor([[x_by_case[key] + shift] for key in case_ids], dtype=torch.float32)
        targets = torch.tensor([[0.25], [0.75]], dtype=torch.float32)[:len(case_ids)]
        return TaskBatch(inputs, None, targets, {}, tuple(case_ids))

    def make_batch(self, case_keys, key):
        assert key.sampling_version == SamplingKey.CASE_EPOCH_VERSION
        shift = float(torch.rand((), dtype=torch.float32).item())
        self.draws.append(shift)
        self.calls += 1
        return self._batch(case_keys, shift)

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        return {"native": sum(batch.targets.numel() for batch in batches)}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        del receivers, execution_mode, phase, epoch, temperature
        return model(scene), {}

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase, auxiliary_state
        return {"native": LossTerm((predictions - targets).square().sum(), targets.numel())}

    def validation_batches(self):
        yield self._batch(self.validation_case_ids)

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        return {
            "squared_error": float((predictions - targets).square().sum()),
            "count": targets.numel(),
        }

    def reduce_native_metrics(self, records):
        return {"field_score": sum(row["squared_error"] for row in records)
                / sum(row["count"] for row in records)}

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        del batch, arm
        return self.loss_terms(predictions, targets, "joint", auxiliary_state)

    def optimizer_groups(self, model, arm, stage):
        assert model is self.model and arm == model.architecture and stage == "joint"
        lr = float(self.optimizer["lr"])
        schedule = ScheduleSpec(
            peak_lr=lr,
            warmup_start_lr=lr,
            warmup_epochs=0,
            hold_through_epoch=self.schedule_horizon - 1,
            total_epochs=self.schedule_horizon,
            final_lr=lr,
        )
        return (OptimizerGroupSpec(
            "joint",
            tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad),
            schedule,
            weight_decay=float(self.optimizer["weight_decay"]),
            betas=tuple(map(float, self.optimizer["betas"])),
            eps=float(self.optimizer["eps"]),
        ),)

    def work_counts(self, batch, predictions, auxiliary_state):
        del predictions, auxiliary_state
        return {"toy_cases": len(batch.case_keys)}

    def training_state_dict(self):
        return {"schema": "r13_6_horizon_toy_provider_state_v1", "calls": self.calls}

    def load_training_state_dict(self, state):
        assert state["schema"] == "r13_6_horizon_toy_provider_state_v1"
        self.calls = int(state["calls"])


def _engine(config, *, horizon: int, checkpoint_epochs: tuple[int, ...]):
    engine_config = _engine_config(
        config, total_epochs=horizon, checkpoint_epochs=checkpoint_epochs,
    )
    return ResearchTrainingEngine(
        engine_config,
        device="cpu",
        parent_age=0,
        selection=SelectionPolicy(field_metric="field_score"),
    )


def _sealed_identity(config, engine, model, provider):
    identity = _run_identity(config, provider)
    provider_identity = dict(provider.identity_payload())
    specs = tuple(provider.optimizer_groups(model, model.architecture, "joint"))
    sealed = {
        **identity,
        "provider_identity": provider_identity,
        "engine_config": _engine_config_payload(engine.config),
        "selection_policy": asdict(engine.selection),
        "optimizer_schedule_contract": [_optimizer_spec_payload(spec) for spec in specs],
    }
    sampler_contract = engine.sampler_epoch_contract()
    if sampler_contract is not None:
        sealed["sampler_epoch_mapping"] = dict(sampler_contract)
    if engine.config.sampling_version == SamplingKey.CASE_EPOCH_VERSION:
        sealed["sampling_dataset_id"] = _sampling_dataset_id(provider_identity)
    return sealed


def _optimizer_step(payload, parameter_name: str) -> int:
    state = payload["optimizer_state_by_name"]["state_by_name"][parameter_name]
    return int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])


@pytest.mark.parametrize(("config_name", "workflow"), CASES)
def test_r13_6_same_lineage_horizon_extension_restores_engine_state(
    tmp_path, monkeypatch, config_name, workflow,
):
    import honf_runtime.unified_training as unified_training

    config, _config_path = load_contextual_frontier_config(CONFIGS / config_name)
    arm = config["member_routing"]["arm"]
    assert config["training"]["optimizer_schedule_horizon"] == 500
    assert config["training"]["curve_every"] == 10
    assert config["training"]["progress_enabled"] is True

    curve_history_lengths = []
    monkeypatch.setattr(
        unified_training,
        "_render_loss_curves",
        lambda history, *_args, **_kwargs: curve_history_lengths.append(len(history)),
    )

    run = tmp_path / arm
    source_model = _TinyModel(arm)
    source_provider = _TinyProvider(source_model, config)
    source_engine = _engine(config, horizon=100, checkpoint_epochs=(100,))
    assert source_engine.config.curve_every == 10
    assert source_engine.config.progress_enabled is True
    source_engine.bind_execution_runtime(
        run / "execution_runtime.json",
        nominal_microbatch_cases=int(config["training"]["microbatch_cases"]),
    )
    torch.manual_seed(13606)
    source_engine.fit(
        source_model, source_provider, run,
        identity=_run_identity(config, source_provider), arm=arm, stop_after=100,
    )

    source_path = run / "checkpoints" / "latest_model.pt"
    source_payload = load_trusted_checkpoint(source_path, map_location="cpu")
    assert source_payload["epoch"] == 100
    assert source_payload["sampler_state"]["next_epoch"] == 101
    assert source_payload["provider_training_state"]["calls"] == 100
    assert _optimizer_step(source_payload, "linear.weight") == 100
    assert source_payload["experiment_identity"]["workflow"] == workflow

    resumed_model = _TinyModel(arm)
    resumed_provider = _TinyProvider(resumed_model, config)
    resumed_engine = _engine(
        config, horizon=500, checkpoint_epochs=(100, 200, 300, 400, 500),
    )
    assert resumed_engine.config.curve_every == 10
    assert resumed_engine.config.progress_enabled is True
    target_identity = _sealed_identity(config, resumed_engine, resumed_model, resumed_provider)
    amendment = _resume_identity_amendment(
        source_payload["experiment_identity"], target_identity,
        allow_horizon_extension=True,
    )
    assert amendment == {
        "kind": "explicit_member_routing_horizon_extension",
        "previous_horizon": 100,
        "execution_horizon": 500,
        "optimizer_schedule_unchanged": True,
        "trajectory_note": (
            "Native objective, constant AdamW rate, provider recipe, data, sampling, and all other run "
            "identity fields remain unchanged."
        ),
    }

    changed_schedule = copy.deepcopy(target_identity)
    changed_schedule["optimizer_schedule_contract"][0]["schedule"]["final_lr"] *= 0.5
    with pytest.raises(ValueError, match="Checkpoint identity"):
        _resume_identity_amendment(
            source_payload["experiment_identity"], changed_schedule,
            allow_horizon_extension=True,
        )
    changed_objective = copy.deepcopy(target_identity)
    changed_objective["recipe"]["native_weights"]["flow/u"] += 0.01
    with pytest.raises(ValueError, match="Checkpoint identity"):
        _resume_identity_amendment(
            source_payload["experiment_identity"], changed_objective,
            allow_horizon_extension=True,
        )

    generator = torch.Generator(device="cpu")
    generator.set_state(source_payload["rng_state"]["torch_cpu"])
    expected_next_draw = float(torch.rand((), generator=generator))

    resumed_engine.bind_execution_runtime(
        run / "execution_runtime.json",
        nominal_microbatch_cases=int(config["training"]["microbatch_cases"]),
    )
    torch.manual_seed(999)
    result = resumed_engine.fit(
        resumed_model, resumed_provider, run,
        identity=_run_identity(config, resumed_provider), arm=arm, stop_after=101,
        resume_checkpoint=source_path, allow_horizon_extension=True,
    )

    resumed_payload = load_trusted_checkpoint(source_path, map_location="cpu")
    assert result["completed_epoch"] == 101
    assert resumed_payload["epoch"] == 101
    assert [row["epoch"] for row in resumed_payload["history"]] == list(range(1, 102))
    assert resumed_payload["sampler_state"]["next_epoch"] == 102
    assert resumed_provider.calls == 101
    assert resumed_provider.sampler_epochs[-1] == 101
    assert resumed_provider.draws == [expected_next_draw]
    assert _optimizer_step(resumed_payload, "linear.weight") == 101
    assert resumed_payload["resume_amendments"][-1]["optimizer_schedule_unchanged"] is True
    assert curve_history_lengths == [1, *range(10, 101, 10), 101]
