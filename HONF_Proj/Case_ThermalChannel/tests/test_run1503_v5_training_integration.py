"""Wrapper and trainer contracts for Run 1503-v5 task-trained detail control."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from channelthermal.interface_field_coupling import (
    TASK_TRAINED_FUNCTIONAL_COALESCENCE,
    _functional_detail_phase_value,
    _functional_detail_ramp_weight,
    _functional_detail_stochastic_mask,
    _phase_functional_probe_kwargs,
)
from channelthermal.training.epoch import assemble_channelthermal_loss_terms
from channelthermal.training.optimizer import build_forward_optimizer
from channelthermal.workflows.train_forward import _resolve_training_loss_config
from honf_forward_core.config import UnifiedForwardConfig
from honf_runtime.config_loader import load_config_bundle
from honf_forward_core.interface_fields.types import PreparedInterfaceField
from honf_forward_core.interface_fields.sparse_incidence_group_control import (
    SparseIncidenceGroupControlPairwiseField,
)
from honf_forward_core.interface_fields.task_trained_functional_coalescence import (
    TaskTrainedFunctionalCoalescencePairwiseField,
)
from torch.nn.parameter import UninitializedParameter


def _wrapper_model(*, epoch: int, training: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            core_honf=SimpleNamespace(
                forward_architecture=TASK_TRAINED_FUNCTIONAL_COALESCENCE
            )
        ),
        core=SimpleNamespace(
            backend=SimpleNamespace(selection_state=lambda: {"epoch": epoch})
        ),
        training=training,
    )


def test_v5_parent_epoch_assignment_does_not_consume_rng() -> None:
    model = _wrapper_model(epoch=50)
    before = torch.get_rng_state().clone()
    mask = _functional_detail_stochastic_mask(model, 5, torch.device("cpu"))
    after = torch.get_rng_state()
    assert mask is None
    assert torch.equal(before, after)


def test_v5_assignment_is_balanced_and_small_batches_alternate() -> None:
    model = _wrapper_model(epoch=51)
    mask = _functional_detail_stochastic_mask(model, 5, torch.device("cpu"))
    assert mask is not None
    assert int(mask.sum()) == 2

    singleton = _wrapper_model(epoch=51)
    sequence = [
        bool(_functional_detail_stochastic_mask(singleton, 1, torch.device("cpu"))[0])
        for _ in range(4)
    ]
    assert sequence == [False, True, False, True]


def test_v5_trainer_ramp_matches_backend_quintic_endpoints() -> None:
    reference = torch.zeros(())
    project_root = Path(__file__).resolve().parents[2]
    profile = json.loads(
        (project_root / "src/config_core/forward/task_trained_functional_coalescence_honf_context.json")
        .read_text(encoding="utf-8")
    )
    backend = TaskTrainedFunctionalCoalescencePairwiseField(
        hidden_dim=8,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=1,
        functional_tree_subsets=profile["model"]["core_honf"]["interface_model"]["functional_tree_subsets"],
    )
    values = []
    for epoch in (50, 51, 100, 150, 500):
        model = _wrapper_model(epoch=epoch)
        wrapper_ramp = float(_functional_detail_ramp_weight(model, reference))
        backend.set_training_progress(epoch=epoch, total_epochs=500)
        backend_ramp = float(backend._ramp(torch.device("cpu"), torch.float32))
        assert wrapper_ramp == backend_ramp
        values.append(wrapper_ramp)
    assert values[0] == 0.0
    assert 0.0 < values[1] < values[2] < 1.0
    assert values[3:] == [1.0, 1.0]


def test_v5_has_no_online_probe_catalogue_even_when_maps_are_requested() -> None:
    model = _wrapper_model(epoch=150)
    result = _phase_functional_probe_kwargs(
        model,
        "P0",
        physical_port_xy=torch.zeros(1, 2, 4, 2),
        current_port_tokens=None,
        module_centers=torch.zeros(1, 2, 2),
        module_present=torch.ones(1, 2),
        ntheta=4,
    )
    assert result == {}


def test_live_phase_complexity_averages_distinct_preparations_once() -> None:
    p0 = PreparedInterfaceField(
        encoded=None,
        module_states=torch.zeros(2, 1),
        backend_state=SimpleNamespace(
            functional_detail_expected_complexity=torch.tensor([0.2, 0.4])
        ),
        coarse_state=torch.zeros(2, 1),
        interaction_aux={},
    )
    p1 = PreparedInterfaceField(
        encoded=None,
        module_states=torch.zeros(2, 1),
        backend_state=SimpleNamespace(
            functional_detail_expected_complexity=torch.tensor([0.8, 0.6])
        ),
        coarse_state=torch.zeros(2, 1),
        interaction_aux={},
    )
    reference = torch.zeros(2, 1)
    value = _functional_detail_phase_value(
        (p0, p1, p0),
        name="functional_detail_expected_complexity",
        batch_size=2,
        reference=reference,
        required=True,
    )
    torch.testing.assert_close(value, torch.tensor([0.5, 0.5]))


def _loss_model(training: bool) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            core_honf=SimpleNamespace(
                forward_architecture=TASK_TRAINED_FUNCTIONAL_COALESCENCE,
                interface_model=None,
            ),
            channelthermal=SimpleNamespace(field_names=["u", "v", "p", "omega", "temperature"]),
        ),
        training=training,
        budgeted_schedule_mode="static",
    )


def _loss_inputs() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    complexity = torch.tensor([0.25, 0.75], requires_grad=True)
    output = {
        "pred_field": torch.zeros(2, 3, 5),
        "organizer_aux": {},
        "functional_detail_expected_complexity": complexity,
        "functional_detail_expected_R": torch.tensor([4.0, 8.0]),
        "functional_detail_ramp_weight": torch.tensor(0.5),
        "functional_detail_stochastic_fraction": torch.tensor(0.5),
        "functional_detail_p0_expected_complexity": torch.tensor([0.25, 0.75]),
        "functional_detail_p0_expected_R": torch.tensor([4.0, 8.0]),
        "functional_detail_p0_actual_R": torch.tensor([12.0, 12.0]),
        "functional_detail_p1_expected_complexity": torch.tensor([0.25, 0.75]),
        "functional_detail_p1_expected_R": torch.tensor([4.0, 8.0]),
        "functional_detail_p1_actual_R": torch.tensor([12.0, 12.0]),
        "functional_detail_p2_expected_complexity": torch.tensor([0.25, 0.75]),
        "functional_detail_p2_expected_R": torch.tensor([4.0, 8.0]),
        "functional_detail_p2_actual_R": torch.tensor([12.0, 12.0]),
    }
    batch = {"field_targets": torch.zeros(2, 3, 5)}
    return output, batch


def test_complexity_is_live_training_loss_but_not_validation_checkpoint_loss() -> None:
    output, batch = _loss_inputs()
    loss_cfg = _resolve_training_loss_config(
        {
            "field_mse_weight": 0.0,
            "organizer_regularization": {"enabled": False},
        },
        {"functional_detail_complexity_weight": 2.0},
    )
    train_terms = assemble_channelthermal_loss_terms(
        output,
        batch,
        _loss_model(training=True),
        loss_cfg,
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        effective_internal_temperature_weight=0.0,
        effective_interface_weight=0.0,
        predicted_consistency_weight=0.0,
    )
    torch.testing.assert_close(train_terms["loss"], torch.tensor(0.5))
    train_terms["loss"].backward()
    torch.testing.assert_close(output["functional_detail_expected_complexity"].grad, torch.tensor([0.5, 0.5]))
    assert train_terms["functional_detail_complexity_weight"].item() == 1.0

    validation_terms = assemble_channelthermal_loss_terms(
        output,
        batch,
        _loss_model(training=False),
        loss_cfg,
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        effective_internal_temperature_weight=0.0,
        effective_interface_weight=0.0,
        predicted_consistency_weight=0.0,
    )
    torch.testing.assert_close(validation_terms["loss"], torch.tensor(0.0))
    torch.testing.assert_close(
        validation_terms["loss_functional_detail_complexity"],
        torch.tensor(0.5),
    )


def test_functional_detail_controller_has_its_own_optimizer_group() -> None:
    model = torch.nn.Module()
    model.core = torch.nn.Module()
    model.core.backend = torch.nn.Module()
    model.core.backend.functional_detail_controller = torch.nn.Linear(3, 2)
    model.core.backend.router = torch.nn.Linear(3, 2)
    optimizer, inventory = build_forward_optimizer(
        model,
        {
            "learning_rate": 3.0e-4,
            "functional_detail_controller_learning_rate": 1.0e-3,
            "organizer_learning_rate": None,
            "weight_decay": 1.0e-5,
        },
    )
    assert inventory["mode"] == "split"
    assert [group["name"] for group in inventory["groups"]] == [
        "functional_detail_controller",
        "prediction",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1.0e-3, 3.0e-4]
    assert all(
        name.startswith("core.backend.functional_detail_controller.")
        for name in inventory["groups"][0]["parameter_names"]
    )


@pytest.mark.parametrize("learning_rate", [0.0, float("nan"), float("inf")])
def test_functional_detail_controller_optimizer_rejects_invalid_lr(learning_rate: float) -> None:
    model = torch.nn.Module()
    model.core = torch.nn.Module()
    model.core.backend = torch.nn.Module()
    model.core.backend.functional_detail_controller = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="functional_detail_controller"):
        build_forward_optimizer(
            model,
            {
                "learning_rate": 3.0e-4,
                "functional_detail_controller_learning_rate": learning_rate,
            },
        )


def test_e50_shared_backend_parameters_and_adamw_updates_match_parent() -> None:
    project_root = Path(__file__).resolve().parents[2]
    profile = json.loads(
        (project_root / "src/config_core/forward/task_trained_functional_coalescence_honf_context.json")
        .read_text(encoding="utf-8")
    )
    interface = profile["model"]["core_honf"]["interface_model"]
    common_args = {
        "hidden_dim": 8,
        "message_hidden_dim": 8,
        "num_heads": 2,
        "fourier_frequencies": 1,
        "group_count": 12,
        "group_control_dim": 16,
        "environment_refinement_normalizer": "sparsemax",
    }
    torch.manual_seed(4321)
    parent_backend = SparseIncidenceGroupControlPairwiseField(**common_args)
    torch.manual_seed(4321)
    candidate_backend = TaskTrainedFunctionalCoalescencePairwiseField(
        **common_args,
        functional_tree_subsets=interface["functional_tree_subsets"],
    )
    candidate_backend.set_training_progress(epoch=50, total_epochs=500)
    parent_parameters = dict(parent_backend.named_parameters())
    candidate_parameters = dict(candidate_backend.named_parameters())
    assert set(parent_parameters) <= set(candidate_parameters)
    for name, parent_parameter in parent_parameters.items():
        candidate_parameter = candidate_parameters[name]
        if isinstance(parent_parameter, UninitializedParameter):
            assert isinstance(candidate_parameter, UninitializedParameter)
            continue
        torch.testing.assert_close(candidate_parameter, parent_parameter, rtol=0.0, atol=0.0)

    class _OptimizerModel(torch.nn.Module):
        def __init__(self, backend: torch.nn.Module) -> None:
            super().__init__()
            self.core = torch.nn.Module()
            self.core.backend = backend

    parent_model = _OptimizerModel(parent_backend)
    candidate_model = _OptimizerModel(candidate_backend)
    parent_optimizer, _ = build_forward_optimizer(
        parent_model,
        {"learning_rate": 3.0e-4, "weight_decay": 1.0e-5},
    )
    candidate_optimizer, _ = build_forward_optimizer(
        candidate_model,
        {
            "learning_rate": 3.0e-4,
            "functional_detail_controller_learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
        },
    )
    candidate_model_parameters = dict(candidate_model.named_parameters())
    for parent_name, parent_parameter in parent_model.named_parameters():
        if parent_name.startswith("core.backend.functional_detail_controller.") or isinstance(
            parent_parameter, UninitializedParameter
        ):
            continue
        gradient = torch.full_like(parent_parameter, 0.125)
        parent_parameter.grad = gradient.clone()
        candidate_model_parameters[parent_name].grad = gradient.clone()
    parent_optimizer.step()
    candidate_optimizer.step()
    for name, parent_parameter in dict(parent_model.named_parameters()).items():
        if name.startswith("core.backend.functional_detail_controller.") or isinstance(
            parent_parameter, UninitializedParameter
        ):
            continue
        torch.testing.assert_close(
            dict(candidate_model.named_parameters())[name],
            parent_parameter,
            rtol=0.0,
            atol=0.0,
        )


def test_e50_prepare_delegates_directly_to_parent_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parents[2]
    profile = json.loads(
        (project_root / "src/config_core/forward/task_trained_functional_coalescence_honf_context.json")
        .read_text(encoding="utf-8")
    )
    interface = profile["model"]["core_honf"]["interface_model"]
    backend = TaskTrainedFunctionalCoalescencePairwiseField(
        hidden_dim=8,
        message_hidden_dim=8,
        num_heads=2,
        fourier_frequencies=1,
        functional_tree_subsets=interface["functional_tree_subsets"],
    )
    backend.set_training_progress(epoch=50, total_epochs=500)
    encoded = object()
    module_states = torch.zeros(2, 12, 8)
    parent_call: dict[str, object] = {}

    def parent_prepare(self, received_encoded, received_states, *, return_routing_maps=False):
        parent_call.update(
            self=self,
            encoded=received_encoded,
            module_states=received_states,
            return_routing_maps=return_routing_maps,
        )
        occupied = torch.zeros(2, 12, dtype=torch.bool)
        occupied[0, :8] = True
        occupied[1, :11] = True
        return {"group_control_state": SimpleNamespace(phase_occupied=occupied)}

    monkeypatch.setattr(SparseIncidenceGroupControlPairwiseField, "prepare", parent_prepare)
    state = backend.prepare(encoded, module_states)
    assert parent_call["self"] is backend
    assert parent_call["encoded"] is encoded
    assert parent_call["module_states"] is module_states
    assert parent_call["return_routing_maps"] is False
    torch.testing.assert_close(state["functional_detail_expected_R"], torch.tensor([8.0, 11.0]))
    torch.testing.assert_close(state["functional_detail_actual_R"], torch.tensor([8.0, 11.0]))
    torch.testing.assert_close(
        state["functional_detail_expected_complexity"], torch.zeros(2)
    )


def test_v5_candidate_profile_is_distinct_and_parent_based() -> None:
    project_root = Path(__file__).resolve().parents[2]
    profile_path = project_root / "src/config_core/forward/task_trained_functional_coalescence_honf_context.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["model"]["core_honf"]["forward_architecture"] == TASK_TRAINED_FUNCTIONAL_COALESCENCE
    assert profile["run"]["id"] == "1507"
    assert profile["training"]["functional_detail_controller_learning_rate"] == 1.0e-3
    assert profile["training"]["functional_detail_complexity_weight"] == pytest.approx(
        0.8382773256398686
    )
    interface = profile["model"]["core_honf"]["interface_model"]
    assert interface["functional_detail_initial_logit"] == 1.6
    assert "read_input_close_rms" not in interface
    assert "read_input_keep_rms" not in interface
    bundle = load_config_bundle(
        "project://src/config_core/forward/task_trained_functional_coalescence_honf_context.json"
    )
    config = UnifiedForwardConfig.from_dict(bundle.effective["model"]["core_honf"])
    assert config.forward_architecture == TASK_TRAINED_FUNCTIONAL_COALESCENCE
    assert bundle.effective["training"]["functional_detail_controller_learning_rate"] == 1.0e-3
    assert bundle.effective["training"]["functional_detail_complexity_weight"] == pytest.approx(
        0.8382773256398686
    )
