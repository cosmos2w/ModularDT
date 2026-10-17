"""Explicit typed capability boundaries and preserved native adapter payloads."""

import pytest
import torch

from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_forward_core.joint_native.model import JointNativeCore
from tests.test_native_budgeted_postfit import _thermal_stats, _thermal_budgeted_config


@pytest.mark.parametrize("mode", ["dense", "joint", "direct"])
def test_thermal_typed_adapter_roundtrip_preserves_normalizer_and_original_state(mode):
    parent = ContentHypergraphCoreV2.from_config({**_thermal_budgeted_config(), "context_depth": 2})
    original = ThermalContentHypergraphAdapterV2(_thermal_stats(), core=parent)
    core = JointNativeCore.from_a2d(parent, routing_mode=mode)
    typed = ThermalContentHypergraphAdapterV2(_thermal_stats(), core=core)
    loaded = ThermalContentHypergraphAdapterV2.from_state_payload(
        typed.state_payload(), state_dict=typed.state_dict(),
    )
    assert typed.normalizer_key == original.normalizer_key == loaded.normalizer_key
    assert typed.get_config()["native_law_contract"] == original.get_config()["native_law_contract"]
    for name, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    assert loaded.core.get_config()["environment_value_reader"] is True


def test_thermal_rejects_relabelling_no_e_core_as_joint_capability():
    config = JointNativeCore.from_a2d(
        ContentHypergraphCoreV2.from_config({**_thermal_budgeted_config(), "context_depth": 2}),
    ).get_config()
    config["environment_value_reader"] = False
    with pytest.raises(ValueError, match="environment_value_reader"):
        ThermalContentHypergraphAdapterV2(_thermal_stats(), core_config=config)


def test_joint_wind_binding_preserves_source_objective_and_dimensions():
    from windfarm.training.content_hypergraph_task import _validate_joint_wind_core_binding

    # A Thermal-shaped core cannot become Wind by merely relabelling the family.
    config = JointNativeCore.from_a2d(
        ContentHypergraphCoreV2.from_config({**_thermal_budgeted_config(), "context_depth": 2}),
    ).get_config()
    recipe = {"schema": "content_hypergraph_recipe_v2", "dataset": "wind",
              "protocol": "source_read_development_v1", "source_reader_variant": "D0",
              "architecture": "A2-D", "seed": 0, "model": {}}
    with pytest.raises(ValueError):
        _validate_joint_wind_core_binding(recipe, config)
