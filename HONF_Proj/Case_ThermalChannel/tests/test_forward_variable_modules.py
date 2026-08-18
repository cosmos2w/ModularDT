from __future__ import annotations

import pytest
import torch

from channelthermal.config import ChannelThermalHONFConfig, ChannelThermalSpecificConfig
from channelthermal.input_adapter import ChannelThermalInputAdapter
from channelthermal.model import ChannelThermalHONFModel
from honf_forward_core.config import UnifiedForwardConfig


def _config() -> ChannelThermalHONFConfig:
    return ChannelThermalHONFConfig(
        core_honf=UnifiedForwardConfig(
            field_dim=5,
            domain_length_x=12.0,
            domain_length_y=6.0,
            coordinate_scale=[12.0, 6.0],
            num_env_tokens_x=4,
            num_env_tokens_y=2,
            num_hyperedges=3,
            hidden_dim=24,
            dropout=0.0,
            decoder_mode="enhanced_honf_pairwise",
            pairwise_kernel_hidden_dim=24,
            boundary_feature_mode="none",
        ),
        channelthermal=ChannelThermalSpecificConfig(
            global_feature_schema="padding_invariant_v2",
            internal_prediction_mode="global_head",
            fallback_hidden_dim=24,
            default_num_interface_points=8,
        ),
    )


def _physical_inputs(width: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    centers = torch.tensor([[[1.0, 1.0], [3.0, 2.0], [7.0, 4.0], [10.0, 5.0]]])
    heat = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    present = torch.ones(1, 4)
    padding = int(width) - 4
    centers = torch.cat([centers, torch.zeros(1, padding, 2)], dim=1)
    heat = torch.cat([heat, torch.zeros(1, padding)], dim=1)
    present = torch.cat([present, torch.zeros(1, padding)], dim=1)
    structure = {
        "re": torch.tensor([[100.0]]),
        "u_in": torch.tensor([[2.0]]),
        "module_centers": centers,
        "heat_powers": heat,
        "module_present": present,
        "material_params": torch.tensor([[1.0e-3, 0.1, 0.01, 10.0, 1.0, 0.45]]),
        "domain_length_x": torch.tensor([[12.0]]),
        "domain_length_y": torch.tensor([[6.0]]),
    }
    query_xy = torch.tensor([[[0.5, 0.5], [4.0, 3.0], [11.0, 5.5]]])
    local_query_points = torch.tensor([[[0.0, 0.0], [0.5, 0.0]]])
    return structure, query_xy, local_query_points


def test_channelthermal_outputs_are_invariant_to_runtime_padding_width() -> None:
    torch.manual_seed(41)
    model = ChannelThermalHONFModel(_config()).eval()
    with torch.no_grad():
        structure_4, query_xy, local_query_points = _physical_inputs(4)
        reference = model(structure_4, query_xy, local_query_points=local_query_points)
        candidates = [
            model(structure, query_xy, local_query_points=local_query_points)
            for structure, _, _ in (_physical_inputs(12), _physical_inputs(32))
        ]

    global_keys = ("pred_field",)
    module_keys = (
        "pred_internal_temperature",
        "pred_interface",
        "pred_port_condition",
        "pred_port_condition_raw",
        "module_response_latent",
    )
    for candidate in candidates:
        for key in global_keys:
            assert torch.allclose(reference[key], candidate[key], atol=1.0e-6, rtol=1.0e-6)
        for key in module_keys:
            assert torch.allclose(reference[key], candidate[key][:, :4], atol=1.0e-6, rtol=1.0e-6)
        assert torch.allclose(
            reference["organizer_aux"]["A_mh"],
            candidate["organizer_aux"]["A_mh"][:, :4],
            atol=1.0e-6,
            rtol=1.0e-6,
        )


def test_padding_invariant_global_features_do_not_change_with_runtime_width() -> None:
    adapter = ChannelThermalInputAdapter(global_feature_schema="padding_invariant_v2")
    contexts = []
    for width in (4, 12, 32):
        structure, _, _ = _physical_inputs(width)
        contexts.append(adapter(**structure).global_context)

    assert adapter.global_context_names[2:6] == (
        "active_module_count",
        "log1p_active_module_count",
        "module_number_density",
        "occupied_area_fraction",
    )
    assert contexts[0].shape[-1] == 18
    assert torch.equal(contexts[0], contexts[1])
    assert torch.equal(contexts[0], contexts[2])


def test_legacy_forward_config_migrates_max_modules_to_fixed_feature_reference() -> None:
    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {"field_dim": 5, "max_num_modules": 12, "hidden_dim": 16},
            "channelthermal": {"internal_prediction_mode": "global_head"},
        }
    )

    assert not hasattr(config.core_honf, "max_num_modules")
    assert config.channelthermal.global_feature_schema == "legacy_v1"
    assert config.channelthermal.legacy_active_fraction_reference_slots == 12

    adapter = ChannelThermalInputAdapter(
        global_feature_schema=config.channelthermal.global_feature_schema,
        legacy_active_fraction_reference_slots=config.channelthermal.legacy_active_fraction_reference_slots,
    )
    contexts = []
    for width in (4, 12, 32):
        structure, _, _ = _physical_inputs(width)
        contexts.append(adapter(**structure).global_context)
    assert contexts[0].shape[-1] == 14
    assert contexts[0][0, 2].item() == pytest.approx(4.0 / 12.0)
    assert torch.equal(contexts[0], contexts[1])
    assert torch.equal(contexts[0], contexts[2])


def test_scheduled_adaptive_selection_is_owned_only_by_final_organizer(monkeypatch: pytest.MonkeyPatch) -> None:
    core = UnifiedForwardConfig(
        field_dim=5,
        domain_length_x=12.0,
        domain_length_y=6.0,
        coordinate_scale=[12.0, 6.0],
        num_env_tokens_x=4,
        num_env_tokens_y=2,
        num_hyperedges=3,
        organizer_mode="exchangeable_slots",
        edge_capacity=4,
        initial_active_edges=4,
        minimum_active_edges=2,
        edge_selection_mode="quality_coverage",
        selection_start_epoch=2,
        selection_transition_epochs=3,
        selection_warmup_mode="all_viable",
        module_assignment_normalizer="scheduled",
        module_sparsity_start_epoch=4,
        module_sparsity_transition_epochs=3,
        environment_assignment_normalizer="scheduled",
        environment_sparsity_start_epoch=4,
        environment_sparsity_transition_epochs=3,
        query_assignment_normalizer="scheduled",
        query_sparsity_start_epoch=3,
        query_sparsity_transition_epochs=2,
        environment_locality_mode="gaussian_bounded",
        query_locality_mode="none",
        hidden_dim=24,
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=24,
        mechanism_state_mode="descriptor_first",
        field_assembly_mode="edge_additive",
        routing_execution="dense",
        boundary_feature_mode="none",
    )
    config = ChannelThermalHONFConfig(
        core_honf=core,
        channelthermal=ChannelThermalSpecificConfig(
            global_feature_schema="padding_invariant_v2",
            internal_prediction_mode="global_head",
            fallback_hidden_dim=24,
            default_num_interface_points=8,
        ),
    )
    model = ChannelThermalHONFModel(config, attach_local_from_checkpoint=False).eval()
    model.set_training_progress(epoch=3, total_epochs=10)
    calls: list[str | None] = []
    original_forward = model.core.organizer.exchangeable.forward

    def recording_forward(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        calls.append(kwargs.get("selection_override"))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.core.organizer.exchangeable, "forward", recording_forward)
    structure, query_xy, local_query_points = _physical_inputs(4)
    with torch.no_grad():
        output = model(structure, query_xy, local_query_points=local_query_points)

    assert calls == ["all", None]
    assert float(output["base_organizer_aux"]["selection_transition_fraction"]) == pytest.approx(1.0 / 3.0)
    assert float(output["organizer_aux"]["selection_transition_fraction"]) == pytest.approx(1.0 / 3.0)
