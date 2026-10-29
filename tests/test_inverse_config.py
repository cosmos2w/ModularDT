"""Strict inverse workflow configuration validation."""

from __future__ import annotations

import pytest

from honf_inverse_core.config import validate_config_keys
from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner


def test_inverse_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown inverse evaluation config keys"):
        validate_config_keys(
            {"request_json": "request.json", "num_palns": 8},
            allowed={"request_json", "num_plans"},
            required={"request_json"},
            label="inverse evaluation",
        )


def test_inverse_config_reports_missing_required_keys() -> None:
    with pytest.raises(ValueError, match="Missing inverse training config keys"):
        validate_config_keys(
            {"device": "cpu"},
            allowed={"device", "dataset_path"},
            required={"device", "dataset_path"},
            label="inverse training",
        )


def test_model_config_rejects_typo_and_bad_matching_mode() -> None:
    with pytest.raises(ValueError, match="Unknown hierarchical inverse model config keys"):
        HierarchicalInverseDesigner.from_config({"num_edges": 3, "plan_layer": 4})
    with pytest.raises(ValueError, match="Unsupported compact-plan matching mode"):
        HierarchicalInverseDesigner.from_config({"num_edges": 3, "matching_mode": "greedy"})


def test_missing_inverse_modes_reconstruct_indexed_ordered_path() -> None:
    designer = HierarchicalInverseDesigner.from_config(
        {"num_edges": 3, "max_modules": 4, "request_hidden_dim": 16}
    )

    assert designer.plan_flow.plan_token_mode == "indexed"
    assert designer.layout_flow.plan_conditioning_mode == "ordered_flat"
    assert designer.model_config["matching_mode"] == "canonical"
    assert "plan_flow.edge_embedding.weight" in designer.state_dict()
    assert "layout_flow.ordered_plan_projection.weight" in designer.state_dict()


def test_exchangeable_inverse_modes_require_schema_provenance_and_sinkhorn() -> None:
    base = {
        "num_edges": 4,
        "max_modules": 5,
        "request_hidden_dim": 16,
        "plan_hidden_dim": 32,
        "layout_hidden_dim": 32,
        "plan_token_mode": "exchangeable_set",
        "plan_conditioning_mode": "set_cross_attention",
        "matching_mode": "sinkhorn",
        "topology_schema_name": "honf_topology_signature",
        "topology_schema_version": 3,
        "forward_topology_checkpoint_sha256": "a" * 64,
        "corrector_enabled": False,
    }
    designer = HierarchicalInverseDesigner.from_config(base)
    assert not any("edge_embedding" in name for name, _ in designer.named_parameters())
    assert not any("ordered_plan_projection" in name for name, _ in designer.named_parameters())

    with pytest.raises(ValueError, match="must be selected together"):
        HierarchicalInverseDesigner.from_config({**base, "plan_conditioning_mode": "ordered_flat"})
    with pytest.raises(ValueError, match="requires matching_mode='sinkhorn'"):
        HierarchicalInverseDesigner.from_config({**base, "matching_mode": "hungarian"})
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        HierarchicalInverseDesigner.from_config({**base, "forward_topology_checkpoint_sha256": ""})
