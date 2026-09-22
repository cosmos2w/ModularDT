"""Focused ThermalChannel contracts for proposed Run 1409."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.data.datasets import (
    CHANNEL_ORDER,
    GLOBAL_INTERFACE_CONDITION_FEATURE_NAMES,
    GLOBAL_INTERFACE_TARGET_NAMES,
)
from channelthermal.local_surrogate.model import LocalModuleConfig, LocalModuleSurrogate
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training_tools.losses import case_group_budget_loss


def test_expected_group_count_is_live_and_reduced_once_per_case() -> None:
    pred = torch.zeros((2, 3, 1), requires_grad=True)
    expected = torch.tensor([2.0, 4.0], requires_grad=True)
    value = case_group_budget_loss(
        {
            "pred_field": pred,
            "case_group_budget_expected_optional_count": expected,
        },
        enabled=True,
        require_live=True,
    )

    assert value.item() == pytest.approx(3.0)
    value.backward()
    torch.testing.assert_close(expected.grad, torch.tensor([0.5, 0.5]))


def test_budget_objective_rejects_detached_aux_fallback() -> None:
    pred = torch.zeros((1, 2, 1))
    with pytest.raises(RuntimeError, match="live top-level"):
        case_group_budget_loss(
            {
                "pred_field": pred,
                "interaction_aux": {
                    "case_group_budget_expected_optional_count": torch.ones(1),
                },
            },
            enabled=True,
            require_live=True,
        )


def test_disabled_budget_objective_preserves_zero_for_legacy_outputs() -> None:
    pred = torch.zeros((1, 2, 1))
    value = case_group_budget_loss({"pred_field": pred}, enabled=False, require_live=True)
    assert value.shape == torch.Size([])
    assert value.item() == pytest.approx(0.0)


def test_case_budget_settings_accepts_numeric_case_weight() -> None:
    # Keep this small config fixture independent of the new core dataclass;
    # the trainer's resolver is intentionally tolerant while checkpoints are
    # migrated to the opt-in profile.
    from channelthermal.training.epoch import _case_group_budget_settings

    model = SimpleNamespace(
        config=SimpleNamespace(
            core_honf=SimpleNamespace(
                forward_architecture="budgeted_group_control_honf",
                interface_model=SimpleNamespace(case_group_budget=SimpleNamespace(enabled=True)),
            )
        )
    )
    enabled, weight = _case_group_budget_settings(model, {"case_group_budget_weight": 0.125})
    assert enabled is True
    assert weight == pytest.approx(0.125)


def _budgeted_model() -> ChannelThermalHONFModel:
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                "forward_architecture": "budgeted_group_control_honf",
                "field_dim": 5,
                "hidden_dim": 16,
                "domain_length_x": 12.0,
                "domain_length_y": 4.0,
                "coordinate_scale": [12.0, 4.0],
                "boundary_feature_mode": "none",
                "num_env_tokens_x": 4,
                "num_env_tokens_y": 3,
                "interface_model": {
                    "message_hidden_dim": 16,
                    "attention_heads": 4,
                    "relative_fourier_frequencies": 2,
                    "receiver_chunk_size": 8,
                    "group_count": 12,
                    "source_normalizer": "entmax15",
                    "query_normalizer": "entmax15",
                    "module_temperature": 1.0,
                    "environment_temperature": 1.0,
                    "query_temperature": 1.0,
                    "group_control_dim": 16,
                    "case_group_budget": {
                        "enabled": True,
                        "gate_hidden_dim": 32,
                        "hard_concrete_temperature": 2.0 / 3.0,
                        "stretch_lower": -0.1,
                        "stretch_upper": 1.1,
                        "initial_optional_open_probability": 0.95,
                        "always_available_group": 0,
                        "normalization": "gate_reference_overlap",
                    },
                },
            },
            "channelthermal": {
                "use_local_surrogate": True,
                "internal_prediction_mode": "local_surrogate",
                "interaction_refinement_steps": 1,
                "default_num_interface_points": 4,
                "local_surrogate_latent_dim": 16,
            },
        }
    )
    model = ChannelThermalHONFModel(config, attach_local_from_checkpoint=False).eval()
    local = LocalModuleSurrogate(
        LocalModuleConfig(
            hidden_dim=16,
            latent_dim=16,
            num_port_latents=2,
            num_heads=4,
            num_layers=1,
            coord_fourier_frequencies=2,
            dropout=0.0,
        )
    )
    model.local_coupling.set_local_surrogate(
        local,
        freeze=True,
        normalization_config={},
        normalization_stats={},
    )
    return model


def test_budgeted_coupling_reuses_only_one_gate_plan_across_phases() -> None:
    torch.manual_seed(1409010)
    model = _budgeted_model()
    captured: list[object] = []
    original_prepare = model.core.prepare

    def capture(*args: object, **kwargs: object):
        captured.append(kwargs.get("phase_shared_state"))
        return original_prepare(*args, **kwargs)

    model.core.prepare = capture  # type: ignore[method-assign]
    output = model(
        query_xy=torch.tensor([[[1.0, 1.0], [6.0, 2.0], [11.0, 3.0]]]),
        re=torch.tensor([[100.0]]),
        u_in=torch.tensor([[1.0]]),
        module_centers=torch.tensor([[[3.0, 1.5], [8.0, 2.5]]]),
        heat_powers=torch.tensor([[1.0, 2.0]]),
        module_present=torch.ones(1, 2),
        material_params=torch.tensor([[0.01, 0.02, 0.03, 1.0, 0.5, 0.45]]),
        local_port_condition_mode="predicted",
        return_routing_maps=True,
        return_prepared_state=True,
    )

    assert captured[0] is None
    assert len(captured) == 3
    assert captured[1] is captured[2]
    assert captured[1] is output["prepared_state"].phase_shared_state
    expected = output["case_group_budget_expected_optional_count"]
    assert expected.shape == (1,)
    assert torch.isfinite(expected).all()
    assert torch.isfinite(output["pred_field"]).all()
    aux = output["interaction_aux"]
    # Both the case-gate summaries and the inherited Run-1406 control
    # summaries are available for an explicit phase-refresh audit.
    assert "initial_port_case_group_budget_sampled_live_count" in aux
    assert "provisional_case_group_budget_gate_reused" in aux
    assert "initial_port_group_control_module_source_projection_rows" in aux
    assert "provisional_group_control_module_source_projection_rows" in aux


def test_budgeted_predicted_port_update_has_live_gate_gradient() -> None:
    from channelthermal.training.epoch import assemble_channelthermal_loss_terms

    torch.manual_seed(1409011)
    model = _budgeted_model()
    model.train()
    output = model(
        query_xy=torch.tensor([[[1.0, 1.0], [6.0, 2.0], [11.0, 3.0]]]),
        re=torch.tensor([[100.0]]),
        u_in=torch.tensor([[1.0]]),
        module_centers=torch.tensor([[[3.0, 1.5], [8.0, 2.5]]]),
        heat_powers=torch.tensor([[1.0, 2.0]]),
        module_present=torch.ones(1, 2),
        material_params=torch.tensor([[0.01, 0.02, 0.03, 1.0, 0.5, 0.45]]),
        local_port_condition_mode="predicted",
    )
    batch = {"field_targets": torch.zeros_like(output["pred_field"])}
    loss_terms = assemble_channelthermal_loss_terms(
        output,
        batch,
        model,
        {
            "field_mse_weight": 1.0,
            "case_group_budget_weight": 0.1,
        },
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        effective_internal_temperature_weight=0.0,
        effective_interface_weight=0.0,
        predicted_consistency_weight=0.0,
    )
    gate_bias = dict(model.named_parameters())["core.backend.router.case_gate.network.net.2.bias"]
    before = gate_bias.detach().clone()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1.0e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    loss_terms["loss"].backward()
    optimizer.step()

    assert loss_terms["loss_group_budget"].requires_grad
    assert torch.isfinite(loss_terms["loss"]).all()
    assert torch.isfinite(output["case_group_budget_expected_optional_count"]).all()
    assert gate_bias.grad is not None and torch.isfinite(gate_bias.grad).all()
    assert torch.count_nonzero(gate_bias.detach() - before).item() > 0


def test_budgeted_checkpoint_resume_restores_architecture_and_state(tmp_path) -> None:
    from channelthermal.training.checkpoints import (
        _restore_rng_state,
        _validate_resume_checkpoint,
        save_checkpoint,
    )
    from honf_runtime.compat import load_trusted_checkpoint, strip_module_prefix

    source = _budgeted_model()
    source.eval()
    forward_kwargs = {
        "query_xy": torch.tensor([[[1.0, 1.0], [6.0, 2.0], [11.0, 3.0]]]),
        "re": torch.tensor([[100.0]]),
        "u_in": torch.tensor([[1.0]]),
        "module_centers": torch.tensor([[[3.0, 1.5], [8.0, 2.5]]]),
        "heat_powers": torch.tensor([[1.0, 2.0]]),
        "module_present": torch.ones(1, 2),
        "material_params": torch.tensor([[0.01, 0.02, 0.03, 1.0, 0.5, 0.45]]),
        "local_port_condition_mode": "predicted",
    }
    with torch.no_grad():
        # Materialize lazy core modules before constructing the optimizer, as
        # the maintained training workflow does for checkpoint operations.
        source(**forward_kwargs)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
    source.train()
    optimizer.zero_grad(set_to_none=True)
    training_output = source(**forward_kwargs)
    training_loss = training_output["pred_field"].square().mean() + 0.1 * training_output[
        "case_group_budget_expected_optional_count"
    ].mean()
    training_loss.backward()
    optimizer.step()
    source.eval()
    with torch.no_grad():
        source_prediction = source(**forward_kwargs)["pred_field"]
    dataset = SimpleNamespace(
        channel_order=list(CHANNEL_ORDER),
        field_dim=5,
        interface_condition_feature_names=list(GLOBAL_INTERFACE_CONDITION_FEATURE_NAMES),
        interface_target_names=list(GLOBAL_INTERFACE_TARGET_NAMES),
        max_num_modules=2,
        selected_module_counts=[2],
        normalizer=SimpleNamespace(stats={}),
    )
    train_config = {
        "dataset": {
            "dataset_id": "synthetic-run1409-test",
            "dataset_schema": "test",
            "dataset_fingerprint": "test-fingerprint",
            "normalize_inputs": False,
            "normalize_targets": False,
        },
        "loss": {"case_group_budget_weight": 0.0},
    }
    path = tmp_path / "epoch_0007_model.pt"
    save_checkpoint(
        path,
        model=source,
        model_config=source.config,
        train_config=train_config,
        dataset=dataset,
        epoch=7,
        best_metric=0.25,
        optimizer=optimizer,
    )

    checkpoint = load_trusted_checkpoint(path, map_location="cpu")
    resumed = _budgeted_model()
    resumed.eval()
    # The maintained workflow materializes lazy core modules from the first
    # training batch before a strict resume load.
    with torch.no_grad():
        resumed(**forward_kwargs)
    _validate_resume_checkpoint(
        checkpoint,
        model=resumed,
        model_config=resumed.config,
        dataset=dataset,
        dataset_config=train_config["dataset"],
    )
    resumed.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1.0e-3)
    saved_optimizer_state = checkpoint.get("optimizer_state_dict")
    assert saved_optimizer_state and saved_optimizer_state.get("state")
    resumed_optimizer.load_state_dict(saved_optimizer_state)
    assert resumed_optimizer.state_dict()["state"]
    assert checkpoint["epoch"] == 7
    assert checkpoint["model_config"]["core_honf"]["forward_architecture"] == (
        "budgeted_group_control_honf"
    )
    for left, right in zip(source.parameters(), resumed.parameters()):
        torch.testing.assert_close(left, right)

    # Resume must restore the stochastic gate stream as well as parameters and
    # optimizer slots.  Rewind to the checkpoint RNG state before each model
    # consumes its next P0 hard-concrete draw.
    source.train()
    resumed.train()
    _restore_rng_state(checkpoint)
    with torch.no_grad():
        source_draw = source(
            **forward_kwargs,
            return_prepared_state=True,
        )["prepared_state"].phase_shared_state.noise
    _restore_rng_state(checkpoint)
    with torch.no_grad():
        resumed_draw = resumed(
            **forward_kwargs,
            return_prepared_state=True,
        )["prepared_state"].phase_shared_state.noise
    assert source_draw is not None and resumed_draw is not None
    torch.testing.assert_close(source_draw, resumed_draw)
    source.eval()
    resumed.eval()
    with torch.no_grad():
        resumed_prediction = resumed(**forward_kwargs)["pred_field"]
    torch.testing.assert_close(source_prediction, resumed_prediction)
