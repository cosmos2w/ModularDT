"""Opt-in typed temperatures and induced-pair-cost objective contracts."""

import pytest
import torch
from channelthermal.training.checkpoints import _partial_initialize_model
from channelthermal.training.epoch import assemble_channelthermal_loss_terms, run_epoch
from channelthermal.training_tools.losses import induced_pair_cost_loss
from torch import nn

from honf_forward_core.config import (
    ROUTING_TYPED_TEMPERATURE_NAMES,
    RoutingIndexConfig,
    RoutingSparsificationConfig,
    UnifiedForwardConfig,
)
from honf_runtime.config_loader import load_config_bundle


def test_sparsification_is_disabled_and_omitted_from_historical_serialization() -> None:
    routing = RoutingIndexConfig.from_dict({})
    assert routing.sparsification == RoutingSparsificationConfig()
    config = UnifiedForwardConfig.from_dict(
        {"forward_architecture": "routed_pairwise_honf", "interface_model": {"routing": {}}}
    )
    serialized = config.to_dict()["interface_model"]["routing"]
    assert "sparsification" not in serialized
    assert "qe_backend" not in serialized

    science_config = UnifiedForwardConfig.from_dict(
        {
            "forward_architecture": "routed_pairwise_honf",
            "interface_model": {
                "routing": {
                    "qe_backend": "torch",
                    "sparsification": {
                        "enabled": True,
                        "learn_typed_temperatures": True,
                    },
                }
            },
        }
    )
    science_routing = science_config.to_dict()["interface_model"]["routing"]
    assert science_routing["qe_backend"] == "torch"


def test_run2002_profile_is_one_explicit_500_epoch_opt_in() -> None:
    bundle = load_config_bundle(
        "project://src/config_core/forward/routing_module_hubs_sparse_cost_context.json"
    )
    routing = bundle.core["model"]["core_honf"]["interface_model"]["routing"]
    assert bundle.core["run"]["id"] == "2002"
    assert bundle.core["training"]["epochs"] == 500
    assert bundle.core["checkpointing"]["save_epoch_milestones"] == [10, 50, 100, 250, 500]
    assert routing["execution"] == "compiled_exact"
    assert routing["qe_backend"] == "torch"
    settings = dict(routing["sparsification"])
    assert settings.pop("cost_weight") > 0.0
    assert settings == {
        "enabled": True,
        "learn_typed_temperatures": True,
        "relative_density_epsilon": 0.05,
        "module_pair_cost": 1.0,
        "environment_pair_cost": 1.0,
    }
    assert bundle.core["training"]["init_checkpoint_path"].endswith(
        "Run_2001_20260916_170829_routed_module_hubs_quickcheck_optimized/epoch_2500_model.pt"
    )


def test_sparsification_validates_exact_science_surface() -> None:
    enabled = RoutingSparsificationConfig(
        enabled=True,
        learn_typed_temperatures=True,
        cost_weight=0.125,
    )
    assert enabled.enabled
    assert enabled.learn_typed_temperatures
    assert ROUTING_TYPED_TEMPERATURE_NAMES == (
        "log_temperature_source_module",
        "log_temperature_source_environment",
        "log_temperature_query_module",
        "log_temperature_query_environment",
    )
    with pytest.raises(ValueError, match="enabled must be true"):
        RoutingSparsificationConfig(learn_typed_temperatures=True)
    with pytest.raises(ValueError, match="relative_density_epsilon"):
        RoutingSparsificationConfig(enabled=True, relative_density_epsilon=0.0)
    with pytest.raises(ValueError, match="cost_weight"):
        RoutingSparsificationConfig(enabled=True, cost_weight=-1.0)
    with pytest.raises(ValueError, match="module_pair_cost"):
        RoutingSparsificationConfig(enabled=True, module_pair_cost=0.0)


def test_induced_pair_cost_uses_all_physical_read_components_and_keeps_gradients() -> None:
    numerator_a = torch.tensor(2.0, requires_grad=True)
    numerator_b = torch.tensor(3.0, requires_grad=True)
    output = {
        "pred_field": torch.zeros(1, 1, 1),
        "interaction_aux": {
            "initial_port_routing_paircost_numerator": numerator_a,
            "initial_port_routing_paircost_denominator": torch.tensor(4.0),
            "routing_paircost_numerator": numerator_b,
            "routing_paircost_denominator": torch.tensor(6.0),
        },
    }
    loss = induced_pair_cost_loss(output, enabled=True, require_components=True)
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()
    torch.testing.assert_close(numerator_a.grad, torch.tensor(0.1))
    torch.testing.assert_close(numerator_b.grad, torch.tensor(0.1))


def test_induced_pair_cost_accepts_per_case_component_vectors() -> None:
    numerator = torch.tensor([1.0, 3.0], requires_grad=True)
    output = {
        "pred_field": torch.zeros(1, 1, 1),
        "interaction_aux": {
            "routing_paircost_numerator": numerator,
            "routing_paircost_denominator": torch.tensor([2.0, 6.0]),
        },
    }
    loss = induced_pair_cost_loss(output, enabled=True, require_components=True)
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()
    torch.testing.assert_close(numerator.grad, torch.tensor([0.125, 0.125]))


def test_induced_pair_cost_disabled_is_differentiable_zero() -> None:
    field = torch.ones(1, 1, 1, requires_grad=True)
    loss = induced_pair_cost_loss({"pred_field": field}, enabled=False)
    assert loss.item() == 0.0
    assert not loss.requires_grad


def test_loss_assembly_exposes_physical_and_paircost_terms_separately() -> None:
    from types import SimpleNamespace

    sparsification = RoutingSparsificationConfig(
        enabled=True,
        learn_typed_temperatures=True,
        cost_weight=0.2,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            channelthermal=SimpleNamespace(field_names=["temperature"]),
            core_honf=SimpleNamespace(
                interface_model=SimpleNamespace(
                    routing=SimpleNamespace(sparsification=sparsification),
                ),
            ),
        ),
    )
    output = {
        "pred_field": torch.ones(1, 2, 1, requires_grad=True),
        "interaction_aux": {
            "routing_paircost_numerator": torch.tensor(2.0, requires_grad=True),
            "routing_paircost_denominator": torch.tensor(4.0),
            "initial_port_routing_paircost_numerator": torch.tensor(3.0, requires_grad=True),
            "initial_port_routing_paircost_denominator": torch.tensor(6.0),
        },
    }
    terms = assemble_channelthermal_loss_terms(
        output,
        {"field_targets": torch.zeros(1, 2, 1)},
        model,
        {"field_mse_weight": 1.0, "temperature_weight": 1.0},
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        effective_internal_temperature_weight=0.0,
        effective_interface_weight=0.0,
        predicted_consistency_weight=0.0,
    )
    torch.testing.assert_close(terms["loss_physical"], torch.tensor(1.0))
    torch.testing.assert_close(terms["loss_paircost"], torch.tensor(0.5))
    torch.testing.assert_close(terms["loss"], torch.tensor(1.1))


def test_science_epoch_metrics_keep_physical_loss_and_four_temperatures() -> None:
    from types import SimpleNamespace

    class TinyRoutedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(
                channelthermal=SimpleNamespace(field_names=["temperature"]),
                core_honf=SimpleNamespace(
                    forward_architecture="routed_pairwise_honf",
                    interface_model=SimpleNamespace(
                        routing=SimpleNamespace(
                            sparsification=RoutingSparsificationConfig(
                                enabled=True,
                                learn_typed_temperatures=True,
                                cost_weight=0.2,
                            )
                        )
                    ),
                ),
            )

        def forward(self, **kwargs):
            batch, queries = kwargs["query_xy"].shape[:2]
            return {
                "pred_field": self.weight.expand(batch, queries, 1),
                "interaction_aux": {
                    "routing_paircost_numerator": self.weight.square(),
                    "routing_paircost_denominator": torch.tensor(2.0),
                    **{
                        "routing_temperature_" + name.removeprefix("log_temperature_"): torch.tensor(1.0)
                        for name in ROUTING_TYPED_TEMPERATURE_NAMES
                    },
                },
            }

    model = TinyRoutedModel()
    batch = {
        "field_targets": torch.zeros(2, 3, 1),
        "query_xy": torch.zeros(2, 3, 2),
        "structure": {},
    }
    metrics = run_epoch(
        model,
        [batch],
        torch.device("cpu"),
        {"field_mse_weight": 1.0, "temperature_weight": 1.0},
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        scaler=None,
        amp=False,
        max_batches=None,
        local_port_condition_mode="predicted",
        mixed_teacher_ratio=0.0,
        effective_internal_temperature_weight=0.0,
        effective_interface_weight=0.0,
        predicted_consistency_weight=0.0,
    )
    torch.testing.assert_close(torch.tensor(metrics["loss_physical"]), torch.tensor(1.0))
    torch.testing.assert_close(torch.tensor(metrics["loss_paircost"]), torch.tensor(0.5))
    for name in ROUTING_TYPED_TEMPERATURE_NAMES:
        assert metrics["temperature_" + name.removeprefix("log_temperature_")] == 1.0


def test_port_global_probe_contributes_distinct_live_paircost_components() -> None:
    from channelthermal.config import ChannelThermalHONFConfig
    from channelthermal.model import ChannelThermalHONFModel

    config = ChannelThermalHONFConfig.from_dict(
        {
            "core_honf": {
                "forward_architecture": "routed_pairwise_honf",
                "field_dim": 5,
                "hidden_dim": 16,
                "domain_length_x": 12.0,
                "domain_length_y": 4.0,
                "coordinate_scale": [12.0, 6.0],
                "module_radius": 0.45,
                "num_env_tokens_x": 2,
                "num_env_tokens_y": 2,
                "boundary_feature_mode": "none",
                "interface_model": {
                    "message_hidden_dim": 16,
                    "attention_heads": 2,
                    "coarse_latent_count": 2,
                    "coarse_blocks": 1,
                    "receiver_chunk_size": 4,
                    "routing": {
                        "descriptor_dim": 8,
                        "router_hidden_dim": 12,
                        "fine_pair_chunk_size": 32,
                        "execution": "compiled_exact",
                        "sparsification": {
                            "enabled": True,
                            "learn_typed_temperatures": True,
                            "cost_weight": 0.1,
                        },
                    },
                },
            },
            "channelthermal": {
                "use_local_surrogate": False,
                "internal_prediction_mode": "global_head",
                "interaction_refinement_steps": 0,
                "default_num_interface_points": 4,
            },
        }
    )
    model = ChannelThermalHONFModel(config).eval()
    output = model(
        query_xy=torch.tensor([[[1.0, 1.0], [6.0, 2.0], [11.0, 3.0]]]),
        re=torch.tensor([[100.0]]),
        u_in=torch.tensor([[1.0]]),
        module_centers=torch.tensor([[[3.0, 1.5], [8.0, 2.5]]]),
        heat_powers=torch.tensor([[1.0, 2.0]]),
        module_present=torch.ones(1, 2),
        material_params=torch.tensor([[0.01, 0.02, 0.03, 1.0, 0.5, 0.45]]),
        return_port_global_consistency=True,
    )
    aux = output["interaction_aux"]
    assert "port_global_routing_paircost_numerator" in aux
    assert "port_global_routing_paircost_denominator" in aux
    assert aux["port_global_routing_paircost_numerator"].requires_grad
    assert float(aux["port_global_routing_paircost_denominator"].sum()) > 0.0


def test_port_paircost_components_mask_padded_receivers_and_keep_live_gradients() -> None:
    from channelthermal.interface_field_coupling import _mask_port_paircost_components

    numerator = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
    denominator = torch.ones(1, 6)
    masked = _mask_port_paircost_components(
        {
            "routing_paircost_numerator": numerator,
            "routing_paircost_denominator": denominator,
            "routing_temperature_source_module": torch.ones(()),
        },
        torch.tensor([[1.0, 0.0, 1.0]]),
        ports=2,
    )

    torch.testing.assert_close(masked["routing_paircost_numerator"], torch.tensor([[0.0, 1.0, 0.0, 0.0, 4.0, 5.0]]))
    torch.testing.assert_close(masked["routing_paircost_denominator"], torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]]))
    masked["routing_paircost_numerator"].sum().backward()
    torch.testing.assert_close(numerator.grad, torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]]))
    torch.testing.assert_close(masked["routing_temperature_source_module"], torch.ones(()))


def test_enabled_routing_core_registers_only_four_zero_log_temperatures() -> None:
    # This assertion deliberately inspects the reusable core state rather than
    # the backend implementation.  It catches accidental duplicate parameter
    # sets when the exact executor is wired into the optional profile.
    from honf_forward_core.interface_fields.core import InterfaceFieldCore

    config = UnifiedForwardConfig.from_dict(
        {
            "field_dim": 5,
            "hidden_dim": 8,
            "forward_architecture": "routed_pairwise_honf",
            "interface_model": {
                "attention_heads": 2,
                "message_hidden_dim": 8,
                "coarse_latent_count": 2,
                "coarse_blocks": 1,
                "routing": {
                    "sparsification": {
                        "enabled": True,
                        "learn_typed_temperatures": True,
                        "cost_weight": 0.125,
                    }
                },
            },
        }
    )
    core = InterfaceFieldCore(config)
    names = tuple(core.routing_log_temperatures)
    assert names == ROUTING_TYPED_TEMPERATURE_NAMES
    assert len(tuple(core.routing_log_temperatures.parameters())) == 4
    assert all(parameter.ndim == 0 for parameter in core.routing_log_temperatures.parameters())
    assert all(float(parameter.detach()) == 0.0 for parameter in core.routing_log_temperatures.parameters())
    assert sum(name.startswith("routing_log_temperatures.") for name, _ in core.named_parameters()) == 4


def test_typed_warmstart_loads_full_state_and_injects_only_four_zero_keys() -> None:
    from types import SimpleNamespace

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                core_honf=SimpleNamespace(
                    interface_model=SimpleNamespace(
                        routing=SimpleNamespace(
                            sparsification=SimpleNamespace(learn_typed_temperatures=True),
                        ),
                    ),
                    field_assembly_mode="context_fusion",
                    organizer_mode="fixed_projection",
                ),
            )
            self.core = nn.Module()
            self.core.shared = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.core.register_buffer("normalization", torch.tensor([3.0, 4.0]))
            self.core.routing_log_temperatures = nn.ParameterDict()
            for name in ROUTING_TYPED_TEMPERATURE_NAMES:
                self.core.routing_log_temperatures[name] = nn.Parameter(torch.full((), 9.0))

    model = DummyModel()
    source_config = SimpleNamespace(
        core_honf=SimpleNamespace(
            interface_model=SimpleNamespace(
                routing=SimpleNamespace(
                    sparsification=SimpleNamespace(learn_typed_temperatures=False),
                ),
            ),
            field_assembly_mode="context_fusion",
            organizer_mode="fixed_projection",
        ),
    )
    source_state = {
        "core.shared": torch.tensor([7.0, 8.0]),
        "core.normalization": torch.tensor([5.0, 6.0]),
    }
    inventory = _partial_initialize_model(
        model,
        {"model_state_dict": source_state},
        source_config=source_config,
    )
    assert inventory["initialized"] == [
        f"core.routing_log_temperatures.{name}"
        for name in ROUTING_TYPED_TEMPERATURE_NAMES
    ]
    assert inventory["missing"] == []
    torch.testing.assert_close(model.core.shared, torch.tensor([7.0, 8.0]))
    torch.testing.assert_close(model.core.normalization, torch.tensor([5.0, 6.0]))
    for parameter in model.core.routing_log_temperatures.values():
        torch.testing.assert_close(parameter, torch.zeros(()))


def test_typed_warmstart_rejects_unrelated_source_state_key() -> None:
    from types import SimpleNamespace

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                core_honf=SimpleNamespace(
                    interface_model=SimpleNamespace(
                        routing=SimpleNamespace(
                            sparsification=SimpleNamespace(learn_typed_temperatures=True),
                        ),
                    ),
                    field_assembly_mode="context_fusion",
                    organizer_mode="fixed_projection",
                ),
            )
            self.core = nn.Module()
            self.core.shared = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.core.routing_log_temperatures = nn.ParameterDict()
            for name in ROUTING_TYPED_TEMPERATURE_NAMES:
                self.core.routing_log_temperatures[name] = nn.Parameter(torch.zeros(()))

    model = DummyModel()
    source_config = SimpleNamespace(
        core_honf=SimpleNamespace(
            interface_model=SimpleNamespace(
                routing=SimpleNamespace(
                    sparsification=SimpleNamespace(learn_typed_temperatures=False),
                ),
            ),
            field_assembly_mode="context_fusion",
            organizer_mode="fixed_projection",
        ),
    )
    with pytest.raises(ValueError, match="differ only by the four new log temperatures"):
        _partial_initialize_model(
            model,
            {
                "model_state_dict": {
                    "core.shared": torch.tensor([7.0, 8.0]),
                    "unexpected": torch.tensor(1.0),
                },
            },
            source_config=source_config,
        )


def test_compiled_maps_merge_unequal_chunks_in_batch_major_order() -> None:
    from honf_forward_core.interface_fields.core import _merge_compiled_routing_maps

    def entry(complete, prior, ptr, source, values):
        return {'routing_module_' + key: torch.tensor(value, dtype=dtype)
                for key, value, dtype in (
                    ('complete_row_index', complete, torch.long),
                    ('complete_prior', prior, torch.float64),
                    ('partial_row_ptr', ptr, torch.long),
                    ('partial_source', source, torch.long),
                    ('partial_prior', values, torch.float64))}
    # Two batches, widths two and one: chunk-local row2 is global row3.
    chunks = [(entry([0, 3], [[.1, .9], [.2, .8]], [0, 0, 1, 2, 2], [1, 0], [.4, .5]), 2),
              (entry([0], [[.3, .7]], [0, 0, 1], [1], [.6]), 1)]
    merged = _merge_compiled_routing_maps(chunks)
    assert merged['routing_module_complete_row_index'].tolist() == [0, 2, 4]
    assert merged['routing_module_partial_row_ptr'].tolist() == [0, 0, 1, 1, 2, 2, 3]
    assert merged['routing_module_partial_source'].tolist() == [1, 0, 1]
    torch.testing.assert_close(merged['routing_module_complete_prior'],
                              torch.tensor([[.1, .9], [.3, .7], [.2, .8]], dtype=torch.float64))
