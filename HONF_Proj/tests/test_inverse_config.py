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
