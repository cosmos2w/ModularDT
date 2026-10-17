from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
for directory in (
    PROJECT / "src",
    PROJECT / "Case_ThermalChannel/src",
    PROJECT / "Case_WindFarm/src",
    PROJECT / "tools",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from channelthermal.content_hypergraph import ThermalInheritedAttentionAdapterV1
from hyperedge_readout_train import _check_round6_training_device_scope, _inspect

from honf_forward_core.inherited_attention.moment_model import MomentHyperedgeCoreV1
from honf_forward_core.inherited_attention.readout_model import HyperedgeReadoutCoreV1
from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.hyperedge_readout_factory import (
    HyperedgeReadoutTrainingEngine,
    _capture_frozen_state,
    _loaded_source_binding,
    _trainable_allowlist,
    _validate_readout_checkpoint,
    load_hyperedge_readout_components,
    make_hyperedge_readout_engine,
    validate_readout_provider_pair,
    verify_frozen_state,
)
from honf_runtime.hyperedge_readout_protocol import (
    READOUT_FAMILY,
    READOUT_PROTOCOL,
    READOUT_WORKFLOW,
    make_hyperedge_readout_recipe,
    validate_hyperedge_readout_recipe,
)
from honf_runtime.hyperedge_readout_protocol import (
    canonical_sha256 as readout_sha256,
)
from honf_runtime.moment_hyperedge_protocol import (
    MOMENT_CORE_SCHEMA,
    MOMENT_FAMILY,
    MOMENT_PROTOCOL,
    MOMENT_WORKFLOW,
    make_moment_hyperedge_recipe,
)
from honf_runtime.moment_hyperedge_protocol import (
    canonical_sha256 as moment_sha256,
)
from honf_runtime.unified_training import (
    EngineConfig,
    LossTerm,
    OptimizerGroupSpec,
    SamplingKey,
    ScheduleSpec,
    SelectionPolicy,
    TaskBatch,
    _sampling_dataset_id,
)

NATIVE_RECIPE_DIR = PROJECT / "src/config_core/forward/inherited_attention"


def test_runtime_source_binding_uses_fixed_inventory_not_import_order(monkeypatch):
    first = _loaded_source_binding(PROJECT)
    injected = ModuleType("round6_import_order_probe")
    injected.__file__ = str(PROJECT / "src/honf_runtime/unified_training.py")
    monkeypatch.setitem(sys.modules, injected.__name__, injected)
    second = _loaded_source_binding(PROJECT)
    assert first == second
    assert first["schema"] == "hyperedge_readout_loaded_source_v1"
    assert "src/honf_runtime/hyperedge_readout_factory.py" in first["files"]
    assert "Case_ThermalChannel/src/channelthermal/content_hypergraph.py" in first["files"]
    assert "Case_WindFarm/src/windfarm/content_hypergraph.py" in first["files"]
    assert "tools/inherited_attention_train.py" in first["files"]
    assert "tools/hyperedge_readout_train.py" in first["files"]
    assert not any(path.startswith(("tests/", "diagnostics/")) for path in first["files"])
    script = (
        "import json,sys; from pathlib import Path; root=Path(sys.argv[1]); "
        "sys.path[:0]=[str(root/'src'),str(root/'Case_ThermalChannel/src'),"
        "str(root/'Case_WindFarm/src'),str(root/'tools')]; "
        "from honf_runtime.hyperedge_readout_factory import _loaded_source_binding; "
        "print(json.dumps(_loaded_source_binding(root),sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(PROJECT)], check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert json.loads(result.stdout) == first


def _thermal_stats() -> dict[str, list[float]]:
    return {
        "field_mean_by_channel": [1.0, -2.0, 0.5, 3.0, 10.0],
        "field_std_by_channel": [2.0, 3.0, 4.0, 5.0, 6.0],
        "interface_targets_mean": [10.0, 0.0],
        "interface_targets_std": [4.0, 5.0],
        "internal_temperature_std": [7.0],
    }


def _native_recipe(dataset: str) -> dict[str, Any]:
    path = NATIVE_RECIPE_DIR / f"{dataset}_iag_c_soft_development_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _round5_source(tmp_path: Path, dataset: str, *, age: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    old_parent = tmp_path / f"{dataset}_c_soft_e500.pt"
    old_parent.write_bytes(f"round5-parent-{dataset}".encode())
    native = _native_recipe(dataset)
    round5_recipe = make_moment_hyperedge_recipe(native, old_parent, "H")
    model_config = {
        "schema": MOMENT_CORE_SCHEMA,
        "architecture": "IAG-C",
        **round5_recipe["model"],
    }
    if dataset == "thermal":
        parent_core = MomentHyperedgeCoreV1.from_config(model_config)
        parent_adapter = ThermalInheritedAttentionAdapterV1(
            _thermal_stats(), core=parent_core, nx=8, ny=6,
        )
        parent_provider = {
            "schema": "thermal_inherited_attention_provider_test_v1",
            "dataset": "ThermalChannel",
            "training_case_ids": ["train-0", "train-1"],
            "model_config": parent_adapter.get_config(),
            "adapter_state": parent_adapter.state_payload(),
        }
    else:
        parent_provider = {
            "schema": "wind_inherited_attention_provider_test_v1",
            "dataset": "WindFarm",
            "training_case_ids": ["row-0", "row-1"],
            "model_config": model_config,
            "adapter_state": {"schema": "wind_adapter_test_v1", "normalizer_sha256": "a" * 64},
        }
    round5_identity = {
        "family": MOMENT_FAMILY,
        "model_family": MOMENT_FAMILY,
        "workflow": MOMENT_WORKFLOW,
        "protocol": MOMENT_PROTOCOL,
        "recipe": round5_recipe,
        "recipe_sha256": moment_sha256(round5_recipe),
        "provider_identity": parent_provider,
    }
    source_path = tmp_path / f"{dataset}_H_e{age}.pt"
    torch.save({
        "workflow": MOMENT_WORKFLOW,
        "arm": "IAG-C",
        "epoch": age,
        "experiment_identity": round5_identity,
    }, source_path)
    return round5_recipe, parent_provider, source_path


def _joint_gate(tmp_path: Path, dataset: str) -> dict[str, Any]:
    specs = (
        ("local_signal", "hyperedge_readout_local_signal_v1", "useful_signal"),
        ("native_law", "hyperedge_readout_native_law_v1", "passed"),
        ("resource_reservation", "hyperedge_readout_resource_reservation_v1", "approved"),
    )
    binding: dict[str, Any] = {
        "schema": "hyperedge_readout_joint_gate_binding_v1",
        "dataset": dataset,
    }
    for name, schema, positive_key in specs:
        receipt_path = tmp_path / f"{dataset}_{name}.json"
        receipt = {"schema": schema, "dataset": dataset, positive_key: True}
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        binding[name] = {
            "path": str(receipt_path.resolve()),
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            positive_key: True,
        }
    return binding


def _readout_recipe(tmp_path: Path, dataset: str = "thermal", *, age: int = 2500,
                    reduction: str = "edge_first", phase: str = "frozen_probe"):
    _round5, _provider, source = _round5_source(tmp_path, dataset, age=age)
    gate = _joint_gate(tmp_path, dataset) if phase == "joint_confirmation" else None
    return make_hyperedge_readout_recipe(
        source,
        receiver_reduction=reduction,
        receiver_access="local_cover",
        phase=phase,
        joint_gate_binding=gate,
    )


@pytest.mark.parametrize(("dataset", "age", "phase"), [
    ("thermal", 2500, "frozen_probe"),
    ("wind", 2500, "frozen_probe"),
    ("thermal", 1000, "joint_confirmation"),
])
def test_round6_recipe_phase_and_source_age_are_sealed(tmp_path: Path, dataset: str, age: int, phase: str):
    recipe = validate_hyperedge_readout_recipe(
        _readout_recipe(tmp_path, dataset, age=age, phase=phase),
    )
    assert recipe["source_epoch"] == age
    assert recipe["phase"] == phase
    assert recipe["initial_review_stop"] == (100 if phase == "frozen_probe" else 2500)
    if phase == "frozen_probe":
        assert recipe["frozen_parameter_policy"] is not None
    else:
        assert recipe["frozen_parameter_policy"] is None
    changed = copy.deepcopy(recipe)
    changed["source_epoch"] = 1000 if phase == "frozen_probe" else 2500
    with pytest.raises(ValueError, match="source epoch"):
        validate_hyperedge_readout_recipe(changed)


def test_joint_recipe_rejects_frozen_fit_policy_and_frozen_review_age(tmp_path: Path):
    recipe = _readout_recipe(tmp_path, "thermal", age=1000, phase="joint_confirmation")
    changed = copy.deepcopy(recipe)
    changed["frozen_parameter_policy"] = {"schema": "round6_frozen_readout_allowlist_v1"}
    with pytest.raises(ValueError, match="frozen fit"):
        validate_hyperedge_readout_recipe(changed)
    changed = copy.deepcopy(recipe)
    changed["initial_review_stop"] = 100
    with pytest.raises(ValueError, match="review stop"):
        validate_hyperedge_readout_recipe(changed)


def test_joint_checkpoint_age_must_start_after_the_e1000_parent(tmp_path: Path):
    recipe = _readout_recipe(tmp_path, "thermal", age=1000, phase="joint_confirmation")
    identity = {
        "family": READOUT_FAMILY,
        "model_family": READOUT_FAMILY,
        "workflow": READOUT_WORKFLOW,
        "recipe": recipe,
        "recipe_sha256": readout_sha256(recipe),
        "phase": "joint_confirmation",
        "receiver_reduction": recipe["receiver_reduction"],
        "receiver_access": recipe["receiver_access"],
    }
    with pytest.raises(ValueError, match="checkpoint age"):
        _validate_readout_checkpoint({
            "workflow": READOUT_WORKFLOW,
            "experiment_identity": identity,
            "epoch": 1000,
        })


def test_cuda0_with_default_visibility_is_rejected_before_cuda_initialization(
    tmp_path: Path, monkeypatch,
):
    recipe = _readout_recipe(tmp_path, "thermal", age=2500)
    deterministic = torch.are_deterministic_algorithms_enabled()
    allow_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    allow_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        assert not torch.cuda.is_initialized()
        with pytest.raises(ValueError, match="physical GPU1/GPU2 UUIDs"):
            _check_round6_training_device_scope(recipe, "cuda:0")
        assert not torch.cuda.is_initialized()
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cuda.matmul.allow_tf32 = allow_matmul_tf32
        torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def test_cuda_inspection_is_rejected_without_a_cli_device_scope():
    with pytest.raises(ValueError, match="CPU-only"):
        _inspect(SimpleNamespace(device="cuda:0", checkpoint=Path("unused.pt")))


def test_frozen_state_digest_detects_signed_zero_mutation():
    model = nn.Linear(2, 1)
    model.bias.requires_grad_(False)
    with torch.no_grad():
        model.bias.fill_(0.0)
    frozen = _capture_frozen_state(model, {"weight"})
    with torch.no_grad():
        model.bias[0] = -0.0
    with pytest.raises(ValueError, match="current_digest"):
        verify_frozen_state(model, frozen, ("weight",))


def test_frozen_allowlist_engine_horizon_and_native_sampler_keys_match(tmp_path: Path):
    recipe = _readout_recipe(tmp_path, "thermal", age=2500)
    engine = make_hyperedge_readout_engine(recipe, device="cpu")
    assert engine.config.total_epochs == 1000
    assert engine.sampler_epoch(1) == 2501
    assert engine.sampler_epoch(500) == 3000
    assert engine.sampler_epoch_contract() == recipe["sampler_epoch_mapping"]

    core = HyperedgeReadoutCoreV1.from_config(recipe["core_config"])
    model = ThermalInheritedAttentionAdapterV1(_thermal_stats(), core=core, nx=8, ny=6)
    allowed = _trainable_allowlist(model, "thermal")
    assert set(allowed) == {
        name for name, _parameter in model.named_parameters()
        if name.startswith((
            "core.receiver_readout.", "core.field_head.", "core.response_phi.",
        ))
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in set(allowed))
    frozen = _capture_frozen_state(model, set(allowed))
    assert verify_frozen_state(model, frozen, allowed)["all_frozen_parameters_and_buffers_bitwise_fixed"]

    class Provider:
        def __init__(self):
            self.keys = []

        def identity_payload(self):
            return {"dataset": "ThermalChannel", "training_case_ids": ["a", "b", "c", "d"]}

        def epoch_cases(self, epoch, seed):
            self.requested_epoch = epoch
            self.seed = seed
            return ("a", "b", "c", "d")

        def make_batch(self, case_keys, key):
            self.keys.append(key)
            return TaskBatch(
                scene_inputs=None,
                receivers=None,
                targets=torch.zeros(len(case_keys), 1),
                auxiliary={"query_sampling_sha256": hashlib.sha256(
                    str(key.seed_for("native-query")).encode(),
                ).hexdigest()},
                case_keys=tuple(case_keys),
            )

    provider = Provider()
    ordered, dataset_id = engine.epoch_case_order(provider, 1)
    assert provider.requested_epoch == 2501
    assert dataset_id == _sampling_dataset_id(provider.identity_payload())
    chunks = engine.prepare_update_microbatches(
        provider, arm="IAG-C", phase="joint", epoch=1, update_index=3,
        update_cases=ordered[:2], sampling_dataset_id=dataset_id,
    )
    assert len(chunks) == 1
    actual = provider.keys[0]
    expected = SamplingKey(
        engine.config.seed, 2501, 3, 0, "joint", "IAG-C",
        sampling_version=engine.config.sampling_version,
        dataset_id=dataset_id,
    )
    assert actual.epoch == expected.epoch
    assert actual.seed_for("native-query") == expected.seed_for("native-query")


class _TinyProvider:
    def __init__(self) -> None:
        self.identity = {"dataset": "tiny", "training_case_ids": ["a", "b", "c"]}
        self.requested_epochs: list[int] = []
        self.training_keys: list[SamplingKey] = []

    def identity_payload(self):
        return copy.deepcopy(self.identity)

    def epoch_cases(self, epoch, seed):
        del seed
        self.requested_epochs.append(epoch)
        return ("a", "b", "c")

    def make_batch(self, case_keys, key):
        if key.stage == "joint":
            self.training_keys.append(key)
        query_hash = hashlib.sha256(
            f"{key.seed_for('tiny-query')}:{','.join(case_keys)}".encode(),
        ).hexdigest()
        return TaskBatch(
            scene_inputs=torch.ones(len(case_keys), 1),
            receivers=None,
            targets=torch.full((len(case_keys), 1), 1.7),
            auxiliary={"query_sampling_sha256": query_hash},
            case_keys=tuple(case_keys),
        )

    def loss_denominators(self, batches, phase, arm):
        del phase, arm
        return {"native": float(sum(batch.targets.numel() for batch in batches))}

    def make_scene(self, scene_inputs):
        return scene_inputs

    def predict_native(self, model, scene, receivers, execution_mode, phase, epoch, temperature):
        del receivers, execution_mode, phase, epoch, temperature
        return model(scene), None

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        del phase, auxiliary_state
        return {"native": LossTerm((predictions - targets).square().sum(), targets.numel())}

    def validation_batches(self):
        return [self.make_batch(("validation",), SamplingKey(7, 2501, 0, 0, "validation", "IAG-C"))]

    def validation_metrics(self, predictions, targets, auxiliary_state):
        del auxiliary_state
        return {"squared_error": float((predictions - targets).square().sum()), "count": targets.numel()}

    def reduce_native_metrics(self, records):
        return {"field_score": sum(row["squared_error"] for row in records)
                / sum(row["count"] for row in records)}

    def optimizer_groups(self, model, arm, stage):
        del model, stage
        if arm != "IAG-C":
            raise ValueError("Tiny runtime test only accepts IAG-C.")
        return (OptimizerGroupSpec(
            "tiny_adamw", ("weight", "bias"),
            ScheduleSpec(peak_lr=3e-4, warmup_start_lr=3e-4, warmup_epochs=0,
                         hold_through_epoch=1, total_epochs=2, final_lr=3e-5),
        ),)

    def work_counts(self, batch, predictions, auxiliary_state):
        del predictions, auxiliary_state
        return {"query_rows": batch.targets.numel()}


def test_exact_save_resume_preserves_sampler_age_and_singleton_final_macro(tmp_path: Path):
    recipe = _readout_recipe(tmp_path, "thermal", age=2500)
    config = EngineConfig(
        seed=7,
        microbatch_cases=1,
        effective_cases=2,
        total_epochs=2,
        monitor_every=1,
        checkpoint_epochs=(1, 2),
        latest_every=1,
        curve_every=None,
        progress_enabled=False,
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        training_mode="joint",
    )
    engine = HyperedgeReadoutTrainingEngine(
        config, device="cpu", selection=SelectionPolicy(field_metric="field_score"), recipe=recipe,
    )
    provider = _TinyProvider()
    model = nn.Linear(1, 1)
    identity = {"test_family": "sampler_offset_fit_loop_v1", "provider_identity": provider.identity_payload()}
    output = tmp_path / "resume-run"
    first = engine.fit(model, provider, output, identity=identity, arm="IAG-C", stop_after=1)
    assert first["completed_epoch"] == 1
    first_path = output / "checkpoints/latest_model.pt"
    first_payload = load_trusted_checkpoint(first_path, map_location="cpu")
    first_row = first_payload["history"][-1]
    assert first_row["case_visits"] == 3
    assert first_row["optimizer_updates"] == 2
    assert first_row["partial_update_cases"] == 1
    assert first_payload["experiment_identity"]["sampler_epoch_mapping"] == recipe["sampler_epoch_mapping"]
    assert first_payload["sampler_state"]["absolute_sampling_epoch"] == 2501

    second = engine.fit(
        model, provider, output, identity=identity, arm="IAG-C", stop_after=2,
        resume_checkpoint=first_path,
    )
    final_payload = load_trusted_checkpoint(output / "checkpoints/latest_model.pt", map_location="cpu")
    assert second["completed_epoch"] == 2
    assert [row["epoch"] for row in final_payload["history"]] == [1, 2]
    assert [row["sampler_epoch"] for row in final_payload["history"]] == [2501, 2502]
    assert sum(row["case_visits"] for row in final_payload["history"]) == 6
    assert sum(row["optimizer_updates"] for row in final_payload["history"]) == 4
    assert all(row["partial_update_cases"] == 1 for row in final_payload["history"])
    assert provider.requested_epochs == [2501, 2502]
    training_keys = [key for key in provider.training_keys if key.stage == "joint"]
    assert [key.epoch for key in training_keys] == [2501, 2501, 2501, 2502, 2502, 2502]
    assert {int(state["step"]) for state in final_payload["optimizer_state_by_name"]["state_by_name"].values()} == {4}


def test_standalone_loader_reconstructs_thermal_adapter_from_checkpoint_metadata(
    tmp_path: Path, monkeypatch,
):
    _round5_recipe, parent_provider, source_path = _round5_source(tmp_path, "thermal", age=2500)
    recipe = make_hyperedge_readout_recipe(source_path, receiver_reduction="edge_first")
    core = HyperedgeReadoutCoreV1.from_config(recipe["core_config"])
    adapter = ThermalInheritedAttentionAdapterV1(_thermal_stats(), core=core, nx=8, ny=6)
    child_provider = copy.deepcopy(parent_provider)
    child_provider["model_config"] = adapter.get_config()
    child_provider["adapter_state"] = adapter.state_payload()
    provider_binding = validate_readout_provider_pair(
        parent_provider, child_provider, dataset="thermal", core_config=recipe["core_config"],
    )
    from honf_runtime import inherited_attention_protocol

    configured_policies = []
    monkeypatch.setattr(
        inherited_attention_protocol,
        "configure_inherited_attention_numerics",
        lambda policy: configured_policies.append(policy),
    )
    original_from_state_payload = ThermalInheritedAttentionAdapterV1.from_state_payload

    def observe_adapter_load(cls, payload, *, device="cpu", state_dict=None):
        assert configured_policies == [recipe["native_recipe"]["numerical_policy"]]
        return original_from_state_payload(payload, device=device, state_dict=state_dict)

    monkeypatch.setattr(
        ThermalInheritedAttentionAdapterV1,
        "from_state_payload",
        classmethod(observe_adapter_load),
    )
    source_payload = load_trusted_checkpoint(source_path, map_location="cpu")
    parent_binding = {
        "schema": "hyperedge_readout_parent_binding_v1",
        "phase": "frozen_probe",
        "path": str(source_path.resolve()),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "epoch": 2500,
        "identity": source_payload["experiment_identity"],
        "recipe_sha256": recipe["source_recipe_sha256"],
        "source_provider_identity_sha256": recipe["source_provider_identity_sha256"],
        "native_provider_binding": provider_binding,
        "common_state_attachment": {"schema": "round6_test_attachment_v1"},
    }
    identity = {
        "family": READOUT_FAMILY,
        "model_family": READOUT_FAMILY,
        "workflow": READOUT_WORKFLOW,
        "protocol": READOUT_PROTOCOL,
        "phase": "frozen_probe",
        "receiver_reduction": "edge_first",
        "receiver_access": "local_cover",
        "recipe": recipe,
        "recipe_sha256": readout_sha256(recipe),
        "sampler_epoch_mapping": recipe["sampler_epoch_mapping"],
        "provider_identity": child_provider,
        "branch_binding": parent_binding,
    }
    checkpoint = tmp_path / "round6_epoch_0001_model.pt"
    torch.save({
        "workflow": READOUT_WORKFLOW,
        "arm": "IAG-C",
        "epoch": 1,
        "experiment_identity": identity,
        "model_state_dict": adapter.state_dict(),
    }, checkpoint)
    loaded = load_hyperedge_readout_components(checkpoint, runtime_root=PROJECT, device="cpu")
    assert configured_policies == [recipe["native_recipe"]["numerical_policy"]]
    assert loaded["checkpoint"]["epoch"] == 1
    assert loaded["recipe"]["phase"] == "frozen_probe"
    assert loaded["model"].get_config()["core"] == recipe["core_config"]
    assert loaded["native_adapter"].get_config()["core"] == recipe["core_config"]
