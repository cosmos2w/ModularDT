from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from channelthermal.training.unified_task import (
    FIXED25_FINGERPRINT,
    THERMAL_GATE_COMPACT_C1_TRANSITION,
    THERMAL_GATE_COMPACT_C1_VERSION,
    THERMAL_TRANSFER_EPOCHS,
    THERMAL_TRANSFER_LR,
    ThermalRefinementTask,
    _development_optimizer_seed,
    _provider_identity_for_transfer_comparison,
    _resolve_development_refinement_parent_binding,
    _validate_development_provider_identity,
    _validate_development_refinement_parent_header,
    _validate_inherited_provider_training_state,
    create_task,
)
from torch import nn


class _Core:
    def __init__(self):
        self.policy = None

    def set_execution(self, **policy):
        self.policy = policy


@pytest.mark.parametrize(
    ("adapter_mode", "core_mode"),
    (
        ("warmup", "all_fine"),
        ("full_detail", "all_fine"),
        ("adaptive_detail", "adaptive"),
        ("all_fine", "all_fine"),
        ("all_base", "all_base"),
        ("nearest", "nearest"),
        ("upstream", "upstream"),
        ("shuffle", "shuffle"),
    ),
)
def test_native_adapter_preserves_refinement_evaluation_modes(adapter_mode, core_mode):
    core = _Core()
    model = SimpleNamespace(core=core)
    ThermalRefinementTask._set_execution(
        None, model, adapter_mode, "hard", 0.25, training=False, threshold=0.6
    )
    assert core.policy == {
        "mode": core_mode,
        "phase": "hard",
        "threshold": 0.6,
        "temperature": 0.25,
        "training_signal": False,
        "gate_version": "hard_v1",
        "gate_transition": THERMAL_GATE_COMPACT_C1_TRANSITION,
    }


def test_native_adapter_forwards_explicit_compact_c1_gate():
    core = _Core()
    model = SimpleNamespace(core=core)
    task = SimpleNamespace(
        gate_version=THERMAL_GATE_COMPACT_C1_VERSION,
        gate_transition=THERMAL_GATE_COMPACT_C1_TRANSITION,
    )
    ThermalRefinementTask._set_execution(
        task, model, "adaptive_detail", "hard", 1.0, training=False
    )
    assert core.policy["gate_version"] == THERMAL_GATE_COMPACT_C1_VERSION
    assert core.policy["gate_transition"] == THERMAL_GATE_COMPACT_C1_TRANSITION


def _minimal_provider(tmp_path: Path, *, gate_version: str = "hard_v1", gate_transition=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent_path = tmp_path / "parent.pt"
    flow_path = tmp_path / "flow.pt"
    parent_path.write_bytes(b"parent")
    flow_path.write_bytes(b"flow")
    recipe = {
        "budget": {},
        "weight_decay": 1.0e-5,
        "calibration": {
            "response_scales": {"fluid": 1.0, "surface": 1.0, "material": 1.0},
            "response_coefficient": 0.0,
            "operator_coefficient": 0.0,
            "calibration_case_ids": [],
        },
        "operator_decision": {"operator_constraint": "not_qualified"},
    }
    return ThermalRefinementTask(
        model=SimpleNamespace(core=SimpleNamespace(refinement_config={})),
        parent_model=None,
        parent={"epoch": 2500, "fit_identity": {"mode": "direct"}},
        parent_path=parent_path,
        flow_path=flow_path,
        atlas_directory=tmp_path,
        training_cases=(),
        validation_cases=(),
        manifest={"manifest_sha256": "fixed25-test"},
        train_families=(),
        development_families=(),
        balances=(),
        optimizer_seed=None,
        device="cpu",
        stats={"field_std_by_channel": np.ones(5, dtype=np.float32)},
        recipe=recipe,
        gate_version=gate_version,
        gate_transition=gate_transition,
    )


def test_thermal_compact_gate_identity_is_opt_in_and_transition_is_required(tmp_path: Path):
    legacy = _minimal_provider(tmp_path)
    assert "gate_version" not in legacy.identity_payload()

    compact_root = tmp_path / "compact"
    compact_root.mkdir()
    compact = _minimal_provider(
        compact_root,
        gate_version=THERMAL_GATE_COMPACT_C1_VERSION,
        gate_transition=THERMAL_GATE_COMPACT_C1_TRANSITION,
    )
    identity = compact.identity_payload()
    assert identity["gate_version"] == THERMAL_GATE_COMPACT_C1_VERSION
    assert identity["gate_transition"] == [0.35, 0.65]

    with pytest.raises(ValueError, match="requires its explicit gate_transition"):
        _minimal_provider(
            tmp_path / "missing",
            gate_version=THERMAL_GATE_COMPACT_C1_VERSION,
        )
    with pytest.raises(ValueError, match=r"sealed to \[0.35, 0.65\]"):
        _minimal_provider(
            tmp_path / "altered",
            gate_version=THERMAL_GATE_COMPACT_C1_VERSION,
            gate_transition=(0.3, 0.7),
        )


def test_parent_native_adapter_all_fine_needs_no_refinement_policy():
    model = SimpleNamespace(core=object())
    ThermalRefinementTask._set_execution(None, model, "full_detail", "hard", 1.0, training=False)


def test_route_control_rejects_unknown_mode():
    model = SimpleNamespace(core=_Core())
    with pytest.raises(ValueError, match="Unknown Thermal refinement evaluation mode"):
        ThermalRefinementTask._set_execution(None, model, "route_search", "hard", 1.0, training=False)


def test_unrefined_parent_rejects_subset_route_controls():
    model = SimpleNamespace(core=object())
    with pytest.raises(TypeError, match="require the opt-in refined source-response core"):
        ThermalRefinementTask._set_execution(None, model, "nearest", "hard", 1.0, training=False)


def _development_parent_payload(path: Path):
    path.write_bytes(b"exact-parent-checkpoint")
    identity = {
        "workflow": "unified_interaction_refinement",
        "development_profile": "fixed25_v1",
        "task": "ThermalChannel",
        "run_id": "thermal_adaptive_refine_20261007",
        "engine_config": {"total_epochs": 2500},
        "provider_identity": {
            "task": "ThermalChannel",
            "dataset_split": "fixed25_v1",
            "manifest_fingerprint": FIXED25_FINGERPRINT,
        },
    }
    checkpoint = {
        "workflow": "unified_interaction_refinement",
        "experiment_identity": identity,
        "arm": "adaptive_detail",
        "epoch": 2500,
        "current_epoch": 2500,
        "model_state_dict": {},
        "optimizer_state_by_name": {},
        "provider_training_state": {},
        "rng_state": {"python": (), "numpy": (), "torch_cpu": torch.zeros(1, dtype=torch.uint8)},
    }
    return checkpoint, hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_parent_path_requires_exact_sha_binding(tmp_path: Path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    with pytest.raises(ValueError, match="supplied together"):
        _resolve_development_refinement_parent_binding({"development_refinement_parent": parent})
    with pytest.raises(ValueError, match="64-character SHA-256"):
        _resolve_development_refinement_parent_binding({
            "development_refinement_parent": parent,
            "development_refinement_parent_sha256": "bad",
        })
    assert _resolve_development_refinement_parent_binding({}) == (None, None)


@pytest.mark.parametrize(
    "invalid_binding",
    ("profile", "task", "run_id", "arm", "epoch", "manifest", "gate", "horizon"),
)
def test_development_parent_header_rejects_formal_or_changed_task_bindings(tmp_path: Path, invalid_binding: str):
    path = tmp_path / "parent.pt"
    checkpoint, digest = _development_parent_payload(path)
    checkpoint = copy.deepcopy(checkpoint)
    identity = checkpoint["experiment_identity"]
    if invalid_binding == "profile":
        identity["development_profile"] = "formal_full_data"
    elif invalid_binding == "task":
        identity["task"] = "WindFarm"
    elif invalid_binding == "run_id":
        identity["run_id"] = "thermal_formal3903"
    elif invalid_binding == "arm":
        checkpoint["arm"] = "full_detail"
    elif invalid_binding == "epoch":
        checkpoint["epoch"] = 2499
    elif invalid_binding == "manifest":
        identity["provider_identity"]["manifest_fingerprint"] = "wrong"
    elif invalid_binding == "gate":
        identity["provider_identity"]["gate_version"] = THERMAL_GATE_COMPACT_C1_VERSION
        identity["provider_identity"]["gate_transition"] = list(THERMAL_GATE_COMPACT_C1_TRANSITION)
    else:
        identity["engine_config"]["total_epochs"] = 5000
    with pytest.raises(ValueError, match="Thermal transfer parent"):
        _validate_development_refinement_parent_header(checkpoint, path=path, expected_sha256=digest)


def test_development_parent_header_rejects_wrong_checkpoint_sha(tmp_path: Path):
    path = tmp_path / "parent.pt"
    checkpoint, digest = _development_parent_payload(path)
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        _validate_development_refinement_parent_header(checkpoint, path=path, expected_sha256="0" * 64)
    _validate_development_refinement_parent_header(checkpoint, path=path, expected_sha256=digest)


@pytest.mark.parametrize(
    "invalid_type",
    ("experiment_identity", "provider_identity", "model_state_dict", "optimizer_state_by_name",
     "provider_training_state", "rng_state"),
)
def test_development_parent_header_rejects_non_mapping_bindings(tmp_path: Path, invalid_type: str):
    path = tmp_path / "parent.pt"
    checkpoint, digest = _development_parent_payload(path)
    checkpoint = copy.deepcopy(checkpoint)
    if invalid_type == "provider_identity":
        checkpoint["experiment_identity"][invalid_type] = []
    else:
        checkpoint[invalid_type] = []
    with pytest.raises(TypeError, match="mapping"):
        _validate_development_refinement_parent_header(checkpoint, path=path, expected_sha256=digest)


def test_development_parent_provider_binding_only_allows_gate_and_lineage_changes():
    source = {
        "dataset_split": "fixed25_v1",
        "manifest_fingerprint": FIXED25_FINGERPRINT,
        "normalization_stats_sha256": "norm-sha",
        "training_response_sources": [{"family_id": "0318", "sha256": "response-sha"}],
        "training_objective": "same-native-response-objective",
    }
    child = {
        **source,
        "development_refinement_parent": {"sha256": "parent-sha"},
        "gate_version": THERMAL_GATE_COMPACT_C1_VERSION,
        "gate_transition": [0.35, 0.65],
    }
    assert _provider_identity_for_transfer_comparison(child) == source
    _validate_development_provider_identity(source, child)
    for key in ("manifest_fingerprint", "normalization_stats_sha256", "training_response_sources", "training_objective"):
        changed = copy.deepcopy(child)
        changed[key] = "changed-binding"
        with pytest.raises(ValueError, match="provider/data/normalizer/response/objective"):
            _validate_development_provider_identity(source, changed)


class _MomentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.values = nn.ParameterList([nn.Parameter(torch.ones(())) for _ in range(65)])


class _MomentProvider:
    def __init__(self, names):
        self.names = names

    def optimizer_groups(self, model, arm, stage):
        del model, arm, stage
        return (
            SimpleNamespace(name="thermal_fine", parameter_names=tuple(self.names[:52])),
            SimpleNamespace(name="thermal_refinement", parameter_names=tuple(self.names[52:])),
        )


def test_development_parent_optimizer_seed_keeps_all_65_named_adamw_moments():
    model = _MomentModel()
    names = [name for name, _ in model.named_parameters()]
    groups = [
        {"name": "thermal_fine", "parameter_names": names[:52]},
        {"name": "thermal_refinement", "parameter_names": names[52:]},
    ]
    states = {
        name: {"step": torch.tensor(9000.0), "exp_avg": torch.zeros_like(parameter),
               "exp_avg_sq": torch.ones_like(parameter)}
        for name, parameter in model.named_parameters()
    }
    checkpoint = {
        "optimizer_state_by_name": {"state_by_name": states, "groups": groups},
        "optimizer_group_names": ["thermal_fine", "thermal_refinement"],
    }
    seed, digest = _development_optimizer_seed(checkpoint, model, _MomentProvider(names))
    assert digest
    assert set(seed["state_by_name"]) == set(names)
    assert len(seed["state_by_name"]) == 65
    assert all(float(state["step"]) == 9000 for state in seed["state_by_name"].values())

    missing = copy.deepcopy(checkpoint)
    del missing["optimizer_state_by_name"]["state_by_name"][names[-1]]
    with pytest.raises(ValueError, match="all 65"):
        _development_optimizer_seed(missing, model, _MomentProvider(names))

    invalid_type = copy.deepcopy(checkpoint)
    invalid_type["optimizer_state_by_name"]["state_by_name"] = []
    with pytest.raises(TypeError, match="must be a mapping"):
        _development_optimizer_seed(invalid_type, model, _MomentProvider(names))

    wrong_shape = copy.deepcopy(checkpoint)
    wrong_shape["optimizer_state_by_name"]["state_by_name"][names[0]]["exp_avg"] = torch.ones(2)
    with pytest.raises(ValueError, match="exp_avg is invalid"):
        _development_optimizer_seed(wrong_shape, model, _MomentProvider(names))


def test_inherited_thermal_provider_state_requires_original_train_calibration():
    state = {
        "temperature_std": 2.0,
        "base_loss_weight": 0.025,
        "expected_work_weight": 0.01,
        "expected_work_calibration": {
            "stage": "thermal_expected_work_calibration",
            "absolute_epoch": 601,
            "arm": "adaptive_detail",
            "coefficient": 0.01,
            "validation_values_read": False,
            "stored_uv_used_for_calibration": False,
        },
    }
    assert _validate_inherited_provider_training_state(
        state, temperature_std=2.0, base_loss_weight=0.025)
    changed = copy.deepcopy(state)
    changed["expected_work_calibration"]["validation_values_read"] = True
    with pytest.raises(ValueError, match="TRAIN-only"):
        _validate_inherited_provider_training_state(
            changed, temperature_std=2.0, base_loss_weight=0.025)
    invalid_type = copy.deepcopy(state)
    invalid_type["expected_work_calibration"] = []
    with pytest.raises(TypeError, match="must be a mapping"):
        _validate_inherited_provider_training_state(
            invalid_type, temperature_std=2.0, base_loss_weight=0.025)


def test_transfer_schedule_is_child_only_and_constant_for_500_epochs(tmp_path: Path):
    provider = _minimal_provider(tmp_path)
    model = nn.Module()
    model.fine = nn.Parameter(torch.ones(()))
    model.core = nn.Module()
    model.core.refinement = nn.Module()
    model.core.refinement.detail = nn.Parameter(torch.ones(()))

    legacy = provider.optimizer_groups(model, "adaptive_detail", "warmup")
    assert [spec.schedule.value(1) for spec in legacy] == [3.0e-6, 3.0e-4]
    provider.development_refinement_parent = {"sha256": "declared-transfer-parent"}
    transferred = provider.optimizer_groups(model, "adaptive_detail", "warmup")
    assert all(spec.schedule.total_epochs == THERMAL_TRANSFER_EPOCHS for spec in transferred)
    assert all(spec.schedule.value(epoch) == THERMAL_TRANSFER_LR for spec in transferred for epoch in (1, 3, 100, 500))


def test_task_factory_rejects_transfer_parent_without_required_sha(tmp_path: Path):
    with pytest.raises(ValueError, match="supplied together"):
        create_task({"development_refinement_parent": tmp_path / "missing.pt"})
