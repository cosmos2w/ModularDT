"""CPU ThermalChannel coupling contract for Run-1409 occupancy control."""

from __future__ import annotations

import torch
from channelthermal.config import ChannelThermalHONFConfig
from channelthermal.local_surrogate.model import LocalModuleConfig, LocalModuleSurrogate
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.training.epoch import assemble_channelthermal_loss_terms

from honf_forward_core.interface_fields.occupancy_group_router import OccupancyGroupPlan


def _model() -> ChannelThermalHONFModel:
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                "forward_architecture": "occupancy_adaptive_group_control_honf",
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


def test_predicted_port_coupling_reuses_only_the_occupancy_plan() -> None:
    torch.manual_seed(1409011)
    model = _model()
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
        return_prepared_state=True,
        return_routing_maps=True,
    )

    assert captured[0] is None
    assert len(captured) == 3
    assert isinstance(captured[1], OccupancyGroupPlan)
    assert captured[1] is captured[2]
    assert captured[1].Kmax == 12
    assert captured[1] is output["prepared_state"].phase_shared_state
    assert torch.isfinite(output["pred_field"]).all()
    assert "case_group_budget_expected_optional_count" not in output
    assert "case_group_budget_continuation" not in output
    assert output["interaction_aux"]["occupancy_group_query_mask"].shape[0] == 1

    terms = assemble_channelthermal_loss_terms(
        output,
        {"field_targets": torch.zeros_like(output["pred_field"])},
        model,
        {"field_mse_weight": 1.0},
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        effective_internal_temperature_weight=0.0,
        effective_interface_weight=0.0,
        predicted_consistency_weight=0.0,
    )
    assert terms["loss_group_budget"].item() == 0.0
    assert terms["case_group_budget_weight"].item() == 0.0
    assert terms["loss_paircost"].item() == 0.0
    assert terms["paircost_weight"].item() == 0.0
    assert terms["loss_organizer"].item() == 0.0
