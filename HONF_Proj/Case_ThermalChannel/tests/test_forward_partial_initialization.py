from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from channelthermal.config import ChannelThermalHONFConfig, ChannelThermalSpecificConfig
from channelthermal.model import ChannelThermalHONFModel
from channelthermal.workflows.train_forward import (
    _partial_initialize_model,
    _validate_initialization_checkpoint,
)
from honf_forward_core.config import UnifiedForwardConfig


FIELD_NAMES = ["u", "v", "p", "omega", "temperature"]


def _config(
    assembly: str,
    mechanism: str,
    *,
    organizer_mode: str = "fixed_projection",
) -> ChannelThermalHONFConfig:
    core = UnifiedForwardConfig(
        field_dim=5,
        domain_length_x=6.0,
        domain_length_y=3.0,
        coordinate_scale=[6.0, 3.0],
        num_env_tokens_x=4,
        num_env_tokens_y=2,
        num_hyperedges=6,
        organizer_mode=organizer_mode,
        edge_capacity=6 if organizer_mode == "exchangeable_slots" else 0,
        initial_active_edges=6,
        minimum_active_edges=1,
        slot_refinement_steps=2,
        edge_selection_mode="all",
        hidden_dim=16,
        dropout=0.0,
        decoder_mode="enhanced_honf_pairwise",
        pairwise_kernel_hidden_dim=16,
        pairwise_kernel_num_layers=2,
        mechanism_state_mode=mechanism,
        field_assembly_mode=assembly,
        routing_execution="dense",
    )
    channel = ChannelThermalSpecificConfig(
        field_names=list(FIELD_NAMES),
        use_local_surrogate=False,
        internal_prediction_mode="global_head",
        interaction_refinement_steps=0,
        fallback_hidden_dim=16,
    )
    return ChannelThermalHONFConfig(core_honf=core, channelthermal=channel)


def _initialize_lazy_parameters(model: ChannelThermalHONFModel) -> None:
    generator = torch.Generator().manual_seed(311)
    with torch.no_grad():
        model(
            query_xy=torch.rand(1, 7, 2, generator=generator) * torch.tensor([6.0, 3.0]),
            re=torch.ones(1, 1),
            u_in=torch.ones(1, 1),
            module_centers=torch.rand(1, 3, 2, generator=generator) * torch.tensor([6.0, 3.0]),
            heat_powers=torch.tensor([[1.0, 0.5, 0.0]]),
            module_present=torch.tensor([[1.0, 1.0, 0.0]]),
            material_params=torch.zeros(1, 6),
        )


def _dataset_and_config() -> tuple[SimpleNamespace, dict[str, object]]:
    stats = {
        "field_mean_by_channel": np.zeros(5, dtype=np.float32),
        "field_std_by_channel": np.ones(5, dtype=np.float32),
    }
    dataset = SimpleNamespace(
        channel_order=list(FIELD_NAMES),
        interface_condition_feature_names=["theta", "cos_theta", "sin_theta", "t_env", "h"],
        interface_target_names=["surface_temperature", "normal_flux"],
        field_dim=5,
        normalizer=SimpleNamespace(stats=stats),
    )
    dataset_config: dict[str, object] = {
        "dataset_id": "fixture_dataset",
        "dataset_schema": "fixture_schema_v1",
        "dataset_fingerprint": "a" * 64,
        "normalize_inputs": False,
        "normalize_targets": True,
    }
    return dataset, dataset_config


def test_context_checkpoint_partial_initialization_loads_only_expected_common_parameters() -> None:
    source_config = _config("context_fusion", "residual_concat")
    target_config = _config("edge_additive", "descriptor_first")
    source = ChannelThermalHONFModel(source_config, attach_local_from_checkpoint=False).eval()
    target = ChannelThermalHONFModel(target_config, attach_local_from_checkpoint=False).eval()
    _initialize_lazy_parameters(source)
    _initialize_lazy_parameters(target)
    with torch.no_grad():
        source.core.decoder.query_to_hyper.weight.fill_(0.25)
        target.core.decoder.query_to_hyper.weight.zero_()

    dataset, dataset_config = _dataset_and_config()
    checkpoint = {
        "checkpoint_schema_version": 1,
        "case_id": "ThermalChannel",
        "model_family": "honf_forward",
        "workflow": "forward",
        "model_config": source_config.to_dict(),
        "model_state_dict": source.state_dict(),
        "channel_order": dataset.channel_order,
        "field_dim": dataset.field_dim,
        "interface_condition_feature_names": dataset.interface_condition_feature_names,
        "interface_target_names": dataset.interface_target_names,
        "dataset_id": dataset_config["dataset_id"],
        "dataset_schema": dataset_config["dataset_schema"],
        "dataset_fingerprint": dataset_config["dataset_fingerprint"],
        "global_normalization_config": {
            "normalize_inputs": False,
            "normalize_targets": True,
        },
        "global_normalization_stats": dataset.normalizer.stats,
        "optimizer_state_dict": {"must_not_load": True},
        "epoch": 99,
        "best_metrics": {"must_not_load": 1.0},
        "rng_state": {"must_not_load": True},
    }

    validated_source = _validate_initialization_checkpoint(
        checkpoint,
        model=target,
        dataset=dataset,
        dataset_config=dataset_config,
    )
    inventory = _partial_initialize_model(
        target,
        checkpoint,
        source_config=validated_source,
    )

    torch.testing.assert_close(
        target.core.decoder.query_to_hyper.weight,
        torch.full_like(target.core.decoder.query_to_hyper.weight, 0.25),
    )
    assert "core.decoder.query_to_hyper.weight" in inventory["loaded"]
    assert "core.decoder.background_head.net.3.weight" in inventory["missing"]
    skipped = {item["key"]: item["reason"] for item in inventory["skipped"]}
    assert skipped["core.decoder.pred_head.net.3.weight"] == "context_to_additive_output_head"
    assert skipped["local_coupling.port_refinement_head.net.net.3.weight"] == "context_to_additive_output_head"
    assert all(name in dict(target.named_parameters()) for name in inventory["loaded"])
    assert not any(name.startswith("core.decoder.pred_head.") for name in inventory["loaded"])


def test_phase1_checkpoint_initializes_exchangeable_bridge_but_skips_fixed_organizer() -> None:
    source_config = _config("edge_additive", "descriptor_first")
    target_config = _config(
        "edge_additive",
        "descriptor_first",
        organizer_mode="exchangeable_slots",
    )
    source = ChannelThermalHONFModel(source_config, attach_local_from_checkpoint=False).eval()
    target = ChannelThermalHONFModel(target_config, attach_local_from_checkpoint=False).eval()
    _initialize_lazy_parameters(source)
    _initialize_lazy_parameters(target)
    with torch.no_grad():
        source.core.decoder.background_head.net[-1].weight.fill_(0.125)
        target.core.decoder.background_head.net[-1].weight.zero_()

    checkpoint = {
        "model_state_dict": source.state_dict(),
        "model_config": source_config.to_dict(),
    }
    inventory = _partial_initialize_model(
        target,
        checkpoint,
        source_config=source_config,
    )

    torch.testing.assert_close(
        target.core.decoder.background_head.net[-1].weight,
        torch.full_like(target.core.decoder.background_head.net[-1].weight, 0.125),
    )
    skipped = {item["key"]: item["reason"] for item in inventory["skipped"]}
    source_organizer_parameters = {
        name for name, _ in source.named_parameters() if name.startswith("core.organizer.")
    }
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    expected_common = {
        name
        for name, value in source_parameters.items()
        if not name.startswith("core.organizer.")
        and name in target_parameters
        and tuple(value.shape) == tuple(target_parameters[name].shape)
    }
    assert source_organizer_parameters
    assert all(skipped[name] == "fixed_projection_organizer" for name in source_organizer_parameters)
    assert not any(name.startswith("core.organizer.") for name in inventory["loaded"])
    assert set(inventory["loaded"]) == expected_common
    assert "core.decoder.background_head.net.3.weight" in inventory["loaded"]
    assert "core.decoder.edge_head.net.3.weight" in inventory["loaded"]
    assert "core.decoder.additive_edge_gate" in inventory["loaded"]
    assert any(name.startswith("core.global_encoder.") for name in inventory["loaded"])
    assert any(name.startswith("local_coupling.") for name in inventory["loaded"])
    assert inventory["source_organizer_mode"] == "fixed_projection"
    assert inventory["target_organizer_mode"] == "exchangeable_slots"
