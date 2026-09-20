from __future__ import annotations

import json
from pathlib import Path

import torch

from channelthermal.environment import ChannelThermalEnvironmentBuilder
from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.interface_fields import HypergraphQuadratureField, InterfaceFieldCore
from honf_forward_core.interface_fields.environment_sampling import RegularGridLayout
from honf_runtime.config_loader import load_config_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_URI = "project://src/config_core/forward/hypergraph_quadrature_honf_context.json"


def _hypergraph_payload() -> dict[str, object]:
    return {
        "forward_architecture": "hypergraph_quadrature_honf",
        "field_dim": 3,
        "hidden_dim": 16,
        "coordinate_scale": [8.0, 4.0],
        "boundary_feature_mode": "none",
        "interface_model": {
            "message_hidden_dim": 16,
            "attention_heads": 4,
            "relative_fourier_frequencies": 2,
            "receiver_chunk_size": 8,
            "group_count": 6,
            "source_normalizer": "entmax15",
            "query_normalizer": "entmax15",
            "module_temperature": 1.0,
            "environment_temperature": 1.0,
            "query_temperature": 1.0,
            "group_control_dim": 16,
            "samples_per_group": 4,
        },
    }


def test_run1408_profile_keeps_run1407_scientific_controls() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "src/config_core/forward/hypergraph_quadrature_honf_context.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = load_config_bundle(PROFILE_URI)
    core = bundle.effective["model"]["core_honf"]
    interface = core["interface_model"]

    assert payload["run"]["id"] == "1408"
    assert core["forward_architecture"] == "hypergraph_quadrature_honf"
    assert interface["group_count"] == 6
    assert interface["group_control_dim"] == 16
    assert interface["samples_per_group"] == 4
    assert bundle.effective["training"]["epochs"] == 50
    assert bundle.effective["checkpointing"]["save_epoch_milestones"] == [
        10,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000,
    ]


def test_run1408_config_requires_fixed_budget_and_serializes_it() -> None:
    config = UnifiedForwardConfig.from_dict(_hypergraph_payload())
    assert config.interface_model.samples_per_group == 4
    assert config.to_dict()["interface_model"]["samples_per_group"] == 4

    invalid = _hypergraph_payload()
    invalid["interface_model"] = dict(invalid["interface_model"])
    invalid["interface_model"]["samples_per_group"] = 3
    try:
        UnifiedForwardConfig.from_dict(invalid)
    except ValueError as exc:
        assert "samples_per_group" in str(exc)
    else:  # pragma: no cover - assertion branch documents the contract.
        raise AssertionError("Run 1408 must keep the initial four-sample budget.")


def test_run1408_factory_uses_opt_in_backend_and_three_term_context() -> None:
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict(_hypergraph_payload()))
    assert isinstance(core.backend, HypergraphQuadratureField)
    assert core.backend.samples_per_group == 4
    assert core.common.__class__.__name__ == "ThreeTermInterfaceContext"


def test_channelthermal_adapter_exposes_explicit_regular_grid_measure() -> None:
    environment = ChannelThermalEnvironmentBuilder()(
        batch_size=2,
        num_env_tokens_x=4,
        num_env_tokens_y=3,
        domain_length_x=8.0,
        domain_length_y=6.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    layout = environment.sampler_layout
    assert isinstance(layout, RegularGridLayout)
    assert layout.grid_shape == (3, 4)
    assert layout.token_to_grid.shape == (12, 2)
    assert environment.env_weights is not None
    assert environment.env_weights.shape == (2, 12)
    torch.testing.assert_close(environment.env_weights[0], layout.weights)
    torch.testing.assert_close(layout.physical_bounds, torch.tensor([[0.0, 0.0], [8.0, 6.0]], dtype=torch.float64))
    assert float(layout.weights.min()) > 0.0


def test_sampler_layout_is_optional_and_forwarded_without_changing_old_batch_shape() -> None:
    generator = torch.Generator().manual_seed(1408)
    batch = BatchData(
        module_centers=torch.rand(1, 2, 2, generator=generator),
        module_present=torch.ones(1, 2),
        module_features=torch.rand(1, 2, 3, generator=generator),
        global_context=torch.rand(1, 4, generator=generator),
        query_xy=torch.rand(1, 3, 2, generator=generator),
        query_time=None,
        target_field=None,
        case_name="run1408-plumbing",
        metadata={},
    )
    assert batch.sampler_layout is None

    layout = ChannelThermalEnvironmentBuilder()(
        batch_size=1,
        num_env_tokens_x=2,
        num_env_tokens_y=2,
        domain_length_x=2.0,
        domain_length_y=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).sampler_layout
    env = ChannelThermalEnvironmentBuilder()(
        batch_size=1,
        num_env_tokens_x=2,
        num_env_tokens_y=2,
        domain_length_x=2.0,
        domain_length_y=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with_layout = BatchData(
        **{**batch.__dict__, "env_coords": env.env_coords, "env_features": env.env_features,
           "env_weights": env.env_weights, "sampler_layout": layout}
    )
    core_config = UnifiedForwardConfig.from_dict({
        "forward_architecture": "dense_pairwise_field",
        "field_dim": 3,
        "hidden_dim": 16,
        "interface_model": {"message_hidden_dim": 16, "attention_heads": 4},
    })
    encoded = InterfaceFieldCore(core_config).encode_case(with_layout)
    assert encoded.sampler_layout is layout
