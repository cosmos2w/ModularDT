"""The new loader needs only its checkpoint and preserves native transforms."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from windfarm.content_hypergraph import WindContentHypergraphAdapter
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline

from honf_forward_core.content_hypergraph import ContentHypergraphCore
from honf_runtime.content_hypergraph import load_content_hypergraph_checkpoint


def payload():
    model = ContentHypergraphCore(module_input_dim=2, environment_input_dim=7, coordinate_dim=3,
                                 query_input_dim=7, operating_input_dim=11, context_dim=16,
                                 num_heads=4, context_depth=2, thermal_response_rank=0)
    adapter = WindContentHypergraphAdapter(
        VelocityNormalizer(np.zeros(3), np.ones(3), np.ones(3)),
        VerticalProfileBaseline(np.array([0.0, 2.0]), np.array([[7.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
                                np.ones(2, dtype=np.int64)),
    )
    recipe = {"schema": "content_hypergraph_recipe_v1", "dataset": "wind", "architecture": "A-H6"}
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    value = {
        "workflow": "content_hypergraph_joint_v1", "arm": "A-H6", "epoch": 100,
        "model_state_dict": model.state_dict(),
        "experiment_identity": {"model_family": "honf_content_hypergraph", "recipe": recipe,
                                "recipe_sha256": digest,
                                "provider_identity": {"model_config": model.get_config(),
                                                      "adapter_state": adapter.state_payload()}},
    }
    return model, adapter, value


def test_standalone_load_has_exact_weights_and_physical_transforms(tmp_path, monkeypatch):
    model, adapter, value = payload()
    path = tmp_path / "new.pt"
    torch.save(value, path)
    # Any attempt to reopen the native field source during loading is a bug.
    import windfarm.training.content_hypergraph_task as native_task
    monkeypatch.setattr(native_task, "WindFarmNativeView", lambda *a, **k: pytest.fail("Dataset access during load"))
    loaded, physical, saved = load_content_hypergraph_checkpoint(path)
    assert saved["epoch"] == 100
    assert physical.state_payload() == adapter.state_payload()
    assert loaded.get_config() == model.get_config()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])


@pytest.mark.parametrize("mutation", ["recipe", "arm", "normalizer", "tensor_inventory", "workflow"])
def test_loader_rejects_mismatched_or_incomplete_native_state(tmp_path, mutation):
    _, _, value = payload()
    value = copy.deepcopy(value)
    if mutation == "recipe":
        value["experiment_identity"]["recipe"]["dataset"] = "thermal"
    elif mutation == "arm":
        value["arm"] = "A-D"
    elif mutation == "normalizer":
        del value["experiment_identity"]["provider_identity"]["adapter_state"]["normalizer"]
    elif mutation == "tensor_inventory":
        value["model_state_dict"].pop(next(iter(value["model_state_dict"])))
    else:
        value["workflow"] = "unified_joint_interaction"
    path = tmp_path / "invalid.pt"
    torch.save(value, path)
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        load_content_hypergraph_checkpoint(path)


def payload_v2():
    from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
    from windfarm.content_hypergraph import WindContentHypergraphAdapterV3
    _, physical, value = payload()
    model = ContentHypergraphCoreV2(module_input_dim=2, environment_input_dim=7, coordinate_dim=3,
                                  query_input_dim=7, operating_input_dim=11, context_dim=16,
                                  local_hidden_dim=16, num_heads=4, context_depth=2, thermal_response_rank=0)
    physical = WindContentHypergraphAdapterV3(physical.normalizer, physical.background_profile)
    recipe = {"schema": "content_hypergraph_recipe_v2", "dataset": "wind", "architecture": "A2-H6"}
    value.update(workflow="content_hypergraph_joint_v2", arm="A2-H6", model_state_dict=model.state_dict())
    value["experiment_identity"].update(
        recipe=recipe,
        recipe_sha256=hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        provider_identity={"model_config": model.get_config(), "adapter_state": physical.state_payload()})
    return model, physical, value


def test_v2_standalone_load_is_strict_and_dataset_free(tmp_path, monkeypatch):
    model, physical, value = payload_v2()
    path = tmp_path / "v2.pt"
    torch.save(value, path)
    import windfarm.training.content_hypergraph_task as native_task
    monkeypatch.setattr(native_task, "WindFarmNativeView", lambda *a, **k: pytest.fail("Dataset read in standalone v2 load"))
    loaded, adapter, _ = load_content_hypergraph_checkpoint(path)
    assert loaded.get_config() == model.get_config()
    assert adapter.state_payload() == physical.state_payload()
    assert all(torch.equal(tensor, loaded.state_dict()[name]) for name, tensor in model.state_dict().items())


@pytest.mark.parametrize("mutation", ["workflow", "core_schema", "adapter_schema"])
def test_v2_loader_rejects_cross_version_reconstruction(tmp_path, mutation):
    _, _, value = payload_v2()
    if mutation == "workflow":
        value["workflow"] = "content_hypergraph_joint_v1"
    elif mutation == "core_schema":
        value["experiment_identity"]["provider_identity"]["model_config"].pop("schema")
    else:
        value["experiment_identity"]["provider_identity"]["adapter_state"]["schema"] = "wind_content_hypergraph_adapter_v2"
    path = tmp_path / "mixed.pt"
    torch.save(value, path)
    with pytest.raises((ValueError, KeyError)):
        load_content_hypergraph_checkpoint(path)
