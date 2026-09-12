"""Candidate profiles inherit their parent experiments' physical policy."""

from copy import deepcopy

import pytest

from honf_forward_core.config import UnifiedForwardConfig
from honf_runtime.config_loader import load_config_bundle


@pytest.mark.parametrize(
    "candidate,parent,architecture,changes",
    [
        (
            "nstage2_hierarchical_regional_context",
            "regional_response_interface_context",
            "hierarchical_regional_honf",
            {"response_tree_opening_interval": [1.0, 2.0], "coarse_module_source": "module_states"},
        ),
        (
            "nstage2_group_mediated_reader_context",
            "sparse_interface_geometry_read_context",
            "sparse_interface_honf",
            {"coarse_module_source": "group_states"},
        ),
    ],
)
def test_nstage2_profiles_preserve_parent_policy(candidate, parent, architecture, changes):
    new = load_config_bundle(f"project://src/config_core/forward/{candidate}.json").effective
    old = load_config_bundle(f"project://src/config_core/forward/{parent}.json").effective
    expected_model = deepcopy(old["model"])
    expected_model["core_honf"]["forward_architecture"] = architecture
    expected_model["core_honf"]["interface_model"].update(changes)
    assert new["model"] == expected_model
    for section in ("training", "checkpointing", "dataset", "normalization", "loss", "case"):
        assert new.get(section) == old.get(section)
    config = UnifiedForwardConfig.from_dict(new["model"]["core_honf"])
    assert config.forward_architecture == architecture


def test_historical_serialization_does_not_add_candidate_options():
    for architecture in ("dense_pairwise_field", "geometry_latent_field", "regional_response_honf"):
        config = UnifiedForwardConfig.from_dict(
            {"forward_architecture": architecture, "hidden_dim": 32, "interface_model": {}}
        )
        assert config.interface_model.coarse_module_source == "module_states"
        options = config.to_dict()["interface_model"]
        assert "coarse_module_source" not in options
        assert "response_tree_opening_interval" not in options


@pytest.mark.parametrize("interval", [[2, 1], [1, 1], [0, 2], [1, float("nan")], [1], [True, 2]])
def test_invalid_tree_opening_interval_is_an_input_error(interval):
    with pytest.raises(ValueError, match="response_tree_opening_interval"):
        UnifiedForwardConfig.from_dict(
            {
                "forward_architecture": "hierarchical_regional_honf",
                "hidden_dim": 32,
                "interface_model": {"response_tree_opening_interval": interval},
            }
        )


def test_group_coarse_source_requires_existing_group_preparation():
    with pytest.raises(ValueError, match="requires sparse_interface_honf"):
        UnifiedForwardConfig.from_dict(
            {
                "forward_architecture": "hierarchical_regional_honf",
                "hidden_dim": 32,
                "interface_model": {"coarse_module_source": "group_states"},
            }
        )
