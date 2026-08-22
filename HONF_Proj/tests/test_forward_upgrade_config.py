from __future__ import annotations

import copy

import pytest
import torch

from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.model import HONFNeuralField
from honf_runtime.config_loader import load_config_bundle


MODE_DEFAULTS = {
    "organizer_mode": "fixed_projection",
    "mechanism_state_mode": "residual_concat",
    "field_assembly_mode": "context_fusion",
    "additive_background_mode": "dense_query_attention",
    "module_assignment_normalizer": "softmax",
    "environment_assignment_normalizer": "softmax",
    "query_assignment_normalizer": "softmax",
    "routing_execution": "dense",
}


def _config_payload() -> dict[str, object]:
    return {
        "field_dim": 2,
        "domain_length_x": 4.0,
        "domain_length_y": 2.0,
        "num_env_tokens_x": 3,
        "num_env_tokens_y": 2,
        "num_hyperedges": 3,
        "hidden_dim": 16,
        "dropout": 0.0,
        "decoder_mode": "enhanced_honf_pairwise",
        "pairwise_kernel_hidden_dim": 16,
    }


def _batch() -> BatchData:
    generator = torch.Generator().manual_seed(29)
    return BatchData(
        module_centers=torch.rand(1, 3, 2, generator=generator),
        module_present=torch.tensor([[1.0, 1.0, 0.0]]),
        module_features=torch.randn(1, 3, 4, generator=generator),
        global_context=torch.randn(1, 3, generator=generator),
        query_xy=torch.rand(1, 7, 2, generator=generator),
        query_time=None,
        target_field=None,
        case_name="configuration-test",
        metadata={},
    )


def _initialized_model(config: UnifiedForwardConfig, *, seed: int) -> HONFNeuralField:
    torch.manual_seed(seed)
    model = HONFNeuralField(config).eval()
    with torch.no_grad():
        model(_batch())
    return model


def test_missing_mode_fields_resolve_to_existing_computation() -> None:
    resolved = UnifiedForwardConfig.from_dict(_config_payload())

    for name, expected in MODE_DEFAULTS.items():
        assert getattr(resolved, name) == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("organizer_mode", "projected_slots"),
        ("mechanism_state_mode", "descriptor_concat"),
        ("field_assembly_mode", "mixed_sum"),
        ("additive_background_mode", "mean_only"),
        ("module_assignment_normalizer", "sparsemax"),
        ("environment_assignment_normalizer", "sparsemax"),
        ("query_assignment_normalizer", "sparsemax"),
        ("routing_execution", "masked_dense"),
    ],
)
def test_unknown_forward_mode_is_rejected(name: str, value: str) -> None:
    payload = _config_payload()
    payload[name] = value
    with pytest.raises(ValueError):
        UnifiedForwardConfig.from_dict(payload)


def test_exchangeable_mode_validates_capacity_and_limits() -> None:
    payload = _config_payload()
    payload.update(organizer_mode="exchangeable_slots", edge_capacity=4, initial_active_edges=5)
    with pytest.raises(ValueError, match="initial_active_edges"):
        UnifiedForwardConfig.from_dict(payload)

    payload.update(initial_active_edges=3, minimum_active_edges=4)
    with pytest.raises(ValueError, match="minimum_active_edges"):
        UnifiedForwardConfig.from_dict(payload)

    payload.update(minimum_active_edges=1, query_module_limit=-1)
    with pytest.raises(ValueError, match="limits"):
        UnifiedForwardConfig.from_dict(payload)

    payload.update(query_module_limit=0, candidate_module_mass_fraction_floor=0.0)
    with pytest.raises(ValueError, match="candidate_module_mass_fraction_floor"):
        UnifiedForwardConfig.from_dict(payload)


def test_mode_complete_profile_selects_upgrade_architecture() -> None:
    bundle = load_config_bundle("project://src/config_core/forward/adaptive_sparse_additive.json")
    core = bundle.effective["model"]["core_honf"]

    assert core["organizer_mode"] == "exchangeable_slots"
    assert core["mechanism_state_mode"] == "descriptor_first"
    assert core["field_assembly_mode"] == "edge_additive"
    assert core["module_assignment_normalizer"] == "scheduled"
    assert core["environment_assignment_normalizer"] == "scheduled"
    assert core["query_assignment_normalizer"] == "scheduled"
    assert core["environment_locality_mode"] == "gaussian_bounded"
    assert core["environment_locality_strength"] == 0.25
    assert core["query_locality_mode"] == "none"
    assert core["locality_radius_cap"] == 3.0
    assert core["candidate_module_mass_fraction_floor"] == 0.01
    assert core["candidate_environment_mass_fraction_floor"] == 0.01
    assert core["routing_execution"] == "dense"
    assert core["additive_edge_gate_init"] == 0.1
    assert core["additive_output_init_std"] == 1.0e-3
    assert core["additive_background_mode"] == "dense_query_attention"
    assert core["topology_signature_enabled"] is True


def test_missing_and_explicit_existing_modes_have_identical_output() -> None:
    payload = _config_payload()
    inferred = UnifiedForwardConfig.from_dict(payload)
    explicit = UnifiedForwardConfig.from_dict({**payload, **MODE_DEFAULTS})
    inferred_model = _initialized_model(inferred, seed=31)
    explicit_model = _initialized_model(explicit, seed=31)

    with torch.no_grad():
        inferred_output = inferred_model(_batch())["pred_field"]
        explicit_output = explicit_model(_batch())["pred_field"]

    torch.testing.assert_close(inferred_output, explicit_output, rtol=0.0, atol=0.0)
    assert inferred_model.state_dict().keys() == explicit_model.state_dict().keys()


def test_missing_mode_checkpoint_state_reconstructs_strictly() -> None:
    payload = _config_payload()
    source = _initialized_model(UnifiedForwardConfig.from_dict(payload), seed=37)
    state = copy.deepcopy(source.state_dict())
    reconstructed = _initialized_model(
        UnifiedForwardConfig.from_dict({**payload, **MODE_DEFAULTS}),
        seed=41,
    )

    incompatible = reconstructed.load_state_dict(state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    with torch.no_grad():
        torch.testing.assert_close(
            source(_batch())["pred_field"],
            reconstructed(_batch())["pred_field"],
            rtol=0.0,
            atol=0.0,
        )


def test_query_locality_strength_defaults_to_historical_environment_strength() -> None:
    payload = {
        **_config_payload(),
        "field_assembly_mode": "edge_additive",
        "mechanism_state_mode": "descriptor_first",
        "query_locality_mode": "gaussian_bounded",
        "environment_locality_strength": 0.37,
    }
    inherited = UnifiedForwardConfig.from_dict(payload)
    explicit = UnifiedForwardConfig.from_dict({**payload, "query_locality_strength": 0.37})
    inherited_model = _initialized_model(inherited, seed=53)
    explicit_model = _initialized_model(explicit, seed=53)

    with torch.no_grad():
        inherited_output = inherited_model(_batch())["pred_field"]
        explicit_output = explicit_model(_batch())["pred_field"]

    torch.testing.assert_close(inherited_output, explicit_output, rtol=0.0, atol=0.0)
    assert inherited_model.state_dict().keys() == explicit_model.state_dict().keys()


def test_query_locality_strength_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="query_locality_strength"):
        UnifiedForwardConfig.from_dict(
            {**_config_payload(), "query_locality_strength": -0.01}
        )
