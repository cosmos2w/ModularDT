from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

from windfarm.content_hypergraph import WindInheritedAttentionAdapterV1
from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline
from windfarm.training.content_hypergraph_task import WindContentHypergraphTask
from windfarm.training.inherited_attention_task import (
    WindInheritedAttentionTaskV1,
    _wind_representation,
)

from honf_forward_core.inherited_attention.model import InheritedAttentionCoreV1
from honf_runtime import content_hypergraph as iag_loader
from honf_runtime import inherited_attention_protocol
from honf_runtime.inherited_attention_protocol import IAG_FAMILY, IAG_PROTOCOL


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _ConfiguredCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = dict(config)
        self.architecture = self._config["architecture"]
        self.weight = nn.Parameter(torch.tensor(1.0))

    def get_config(self):
        return dict(self._config)


def _adapter_state():
    adapter = WindInheritedAttentionAdapterV1(
        VelocityNormalizer(np.zeros(3), np.ones(3), np.ones(3)),
        VerticalProfileBaseline(
            np.asarray([0.0, 1.0]), np.zeros((2, 3)), np.ones(2, dtype=np.int64),
        ),
    )
    return adapter.state_payload()


def _recipe(*, sparse: bool = False):
    normalization = "measure_entmax15" if sparse else "soft"
    return {
        "schema": "inherited_attention_recipe_v1",
        "family": IAG_FAMILY,
        "model_family": IAG_FAMILY,
        "protocol": IAG_PROTOCOL,
        "dataset": "wind",
        "architecture": "IAG-Q",
        "representation": "iag_q",
        "normalization": normalization,
        "numerical_policy": "fp32_deterministic_cuda_v1",
        "trajectory": "e500_sparse_child" if sparse else "fresh_soft",
        "model": {
            "module_input_dim": 2,
            "environment_input_dim": 7,
            "coordinate_dim": 3,
            "query_input_dim": 7,
            "operating_input_dim": 11,
            "output_dim": 3,
            "context_dim": 8,
            "num_heads": 2,
            "context_depth": 1,
            "local_hidden_dim": 8,
            "dropout": 0.0,
            "init_seed": 0,
            "signed_carrier_width": 8,
            "thermal_response": False,
            "normalization": normalization,
            "scene_executor": "cached_tokens",
            "query_executor": "cell_selected",
        },
    }


def _review(tmp_path):
    return {
        "schema": "inherited_attention_representation_review_v1",
        "path": str((tmp_path / "review.json").resolve()),
        "sha256": "a" * 64,
        "dataset": "wind",
        "decision": "replace_iag_c_with_iag_q",
        "fresh_run_dir": str((tmp_path / "wind_iag_q").resolve()),
    }


def _checkpoint(*, sparse: bool, tmp_path: Path):
    recipe = _recipe(sparse=sparse)
    review = _review(tmp_path)
    config = {
        "schema": "inherited_attention_core_v1",
        "architecture": "IAG-Q",
        **recipe["model"],
    }
    parent_identity = None
    branch_parent = None
    if sparse:
        parent_recipe = _recipe(sparse=False)
        parent_recipe_sha = _sha256(parent_recipe)
        parent_config = {
            **config,
            "normalization": "soft",
        }
        parent_identity = {
            "workflow": "inherited_attention_joint_v1",
            "model_family": IAG_FAMILY,
            "recipe": parent_recipe,
            "recipe_sha256": parent_recipe_sha,
            "provider_identity": {
                "schema": "wind_inherited_attention_provider_v1",
                "architecture": "IAG-Q",
                "representation": "iag_q",
                "model_config": parent_config,
                "adapter_state": _adapter_state(),
            },
            "representation_review": review,
        }
        parent_path = str((tmp_path / "parent_e0500.pt").resolve())
        source_checkpoint_sha = "b" * 64
        contract = {
            "schema": "inherited_attention_sparse_child_v1",
            "trajectory": "soft_e1_500_then_absolute_sparse_transition_e501_800",
            "source_checkpoint": parent_path,
            "source_checkpoint_sha256": source_checkpoint_sha,
            "source_recipe_sha256": parent_recipe_sha,
            "target_recipe_sha256": _sha256(recipe),
            "source_epoch": 500,
            "source_architecture": "IAG-Q",
            "source_representation": "iag_q",
            "target_architecture": "IAG-Q",
            "target_representation": "iag_q",
            "exact_parameter_layout_verified": True,
            "representation_review_sha256": review["sha256"],
            "scene_executor_transition": {
                "source": "cached_tokens",
                "target": "cached_tokens",
                "same_probability_function": True,
            },
        }
        branch_parent = {
            "path": parent_path,
            "sha256": source_checkpoint_sha,
            "epoch": 500,
            "identity": parent_identity,
            "inherited_attention_sparse_child": contract,
        }
    provider = {
        "schema": "wind_inherited_attention_provider_v1",
        "architecture": "IAG-Q",
        "representation": "iag_q",
        "model_config": config,
        "adapter_state": _adapter_state(),
    }
    identity = {
        "workflow": "inherited_attention_joint_v1",
        "model_family": IAG_FAMILY,
        "recipe": recipe,
        "recipe_sha256": _sha256(recipe),
        "provider_identity": provider,
        "representation_review": review,
    }
    value = {
        "workflow": "inherited_attention_joint_v1",
        "arm": "IAG-Q",
        "epoch": 501 if sparse else 100,
        "experiment_identity": identity,
        "model_state_dict": {"weight": torch.tensor(1.0)},
    }
    if branch_parent is not None:
        value["branch_parent"] = branch_parent
    return value


@pytest.fixture
def stub_q_loader(monkeypatch):
    monkeypatch.setattr(
        inherited_attention_protocol,
        "validate_inherited_attention_recipe",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        inherited_attention_protocol, "configure_inherited_attention_numerics", lambda policy: None,
    )
    monkeypatch.setattr(
        InheritedAttentionCoreV1,
        "from_config",
        classmethod(lambda cls, config: _ConfiguredCore(config)),
    )


@pytest.mark.parametrize(("architecture", "representation"), [("IAG-C", "iag_c"), ("IAG-Q", "iag_q")])
def test_wind_recipe_representation_pairs_are_explicit(architecture, representation):
    assert _wind_representation({"architecture": architecture, "representation": representation}) == (
        architecture, representation,
    )
    with pytest.raises(ValueError, match="architecture and representation"):
        _wind_representation({"architecture": architecture, "representation": "iag_q" if architecture == "IAG-C" else "iag_c"})


def test_wind_q_optimizer_dispatch_is_bound_to_the_recipe_core_without_constructing_optimizer():
    provider = object.__new__(WindInheritedAttentionTaskV1)
    core = _ConfiguredCore({"architecture": "IAG-Q"})
    provider.model = core
    provider.architecture = "IAG-Q"
    provider.recipe = {
        "optimizer": {"lr": 3.0e-4, "weight_decay": 1.0e-5, "betas": [0.9, 0.999], "eps": 1.0e-8},
    }
    provider.total_epochs = 2500

    groups = provider.optimizer_groups(core, "IAG-Q", "joint")

    assert len(groups) == 1
    assert groups[0].parameter_names == ("weight",)
    with pytest.raises(ValueError, match="bound only"):
        provider.optimizer_groups(core, "IAG-C", "joint")
    wrong_core = _ConfiguredCore({"architecture": "IAG-C"})
    with pytest.raises(ValueError, match="bound only"):
        provider.optimizer_groups(wrong_core, "IAG-Q", "joint")


def test_wind_q_prediction_rejects_a_mismatched_execution_mode_before_scene_use():
    provider = object.__new__(WindInheritedAttentionTaskV1)
    core = _ConfiguredCore({"architecture": "IAG-Q"})
    provider.model = core
    provider.architecture = "IAG-Q"

    with pytest.raises(ValueError, match="bound recipe architecture"):
        provider.predict_native(core, None, None, "IAG-C")


def test_wind_q_provider_uses_the_existing_native_scene_query_and_physical_facade():
    from honf_forward_core.content_hypergraph.contracts import QueryBatch, SceneBatch
    from honf_forward_core.inherited_attention import (
        InheritedAttentionCoreV1 as NativeCore,
    )
    from honf_forward_core.inherited_attention import (
        smoothstep_grid_receiver_cover,
        with_receiver_cover,
    )

    provider = object.__new__(WindInheritedAttentionTaskV1)
    anchors = torch.cartesian_prod(*(torch.tensor([0.0, 1.0]) for _ in range(3)))[None]
    scene = SceneBatch(
        module_features=torch.ones(1, 2, 2), environment_features=torch.ones(1, 8, 7),
        module_coordinates=torch.tensor([[[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]]),
        environment_coordinates=anchors,
        module_mask=torch.ones(1, 2, dtype=torch.bool), environment_mask=torch.ones(1, 8, dtype=torch.bool),
        module_measure=torch.ones(1, 2), environment_measure=torch.ones(1, 8),
        module_ids=torch.tensor([[13, 29]]), environment_ids=torch.arange(8)[None],
        local_radius=torch.ones(1, 2), operating_features=torch.ones(1, 11),
        coordinate_origin=torch.zeros(1, 3), coordinate_scale=torch.ones(1, 3),
        module_measure_reference=torch.ones(1), environment_measure_reference=torch.ones(1),
        scene_key="wind-iag-q-provider-fixture", normalizer_key="test", coordinate_frame_key="native",
        module_characteristic_lengths=torch.full((1, 2, 3), 0.2),
        environment_characteristic_lengths=torch.ones(1, 8, 3),
    )
    base_query = QueryBatch(
        coordinates=torch.tensor([[[0.3, 0.4, 0.5], [0.7, 0.6, 0.5]]]),
        features=torch.ones(1, 2, 7), query_mask=torch.ones(1, 2, dtype=torch.bool),
        query_ids=torch.tensor([[41, 43]]), scene_key=scene.scene_key,
        coordinate_frame_key=scene.coordinate_frame_key, receiver_key="q-fixture",
    )
    cover = smoothstep_grid_receiver_cover(
        base_query.coordinates, anchors, scene.environment_ids, scene.environment_mask,
        base_query.query_ids, frame_key=scene.coordinate_frame_key,
        query_mask=base_query.query_mask,
        legal_bounds=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]),
    )
    query = with_receiver_cover(base_query, cover)
    provider.adapter = type("Adapter", (), {
        "make_query_batch": staticmethod(lambda prepared_scene, receivers: query),
        "physical_velocity": staticmethod(
            lambda residual, coordinates: (residual, torch.zeros_like(residual), residual),
        ),
    })()
    core = NativeCore(
        module_input_dim=2, environment_input_dim=7, coordinate_dim=3, query_input_dim=7,
        operating_input_dim=11, context_dim=8, num_heads=2, context_depth=1,
        local_hidden_dim=8, architecture="IAG-Q", scene_executor="cached_tokens",
        query_executor="cell_selected",
    ).eval()
    provider.model = core
    provider.architecture = "IAG-Q"
    receivers = type("Receivers", (), {"role_ids": np.zeros((1, 2), dtype=np.int8)})()

    with torch.no_grad():
        predictions, auxiliary = provider.predict_native(core, scene, receivers, "IAG-Q", epoch=1)

    assert predictions.main_mps.shape == (1, 2, 3)
    assert torch.isfinite(predictions.main_mps).all()
    assert auxiliary["organization"].scene is scene


def test_wind_q_provider_identity_and_serialization_name_the_live_representation(monkeypatch):
    config = {"schema": "inherited_attention_core_v1", "architecture": "IAG-Q"}
    provider = object.__new__(WindInheritedAttentionTaskV1)
    provider.model = _ConfiguredCore(config)
    provider.architecture = "IAG-Q"
    provider.representation = "iag_q"
    provider.recipe = {
        **_recipe(),
        "seed": 0,
        "sampling_seed": 42,
        "optimizer": {"lr": 3.0e-4, "weight_decay": 1.0e-5, "betas": [0.9, 0.999], "eps": 1.0e-8},
    }
    provider.initialization_seed = 0
    provider.sampling_seed = 42
    provider.seed = 42
    provider.role_query_counts = {"volume": 6144, "hub_band": 2048}
    provider.validation_role_query_counts = {"volume": 8192, "hub_band": 2048}
    provider.sampling_reference_scope = {"protocol": "source_read_development_v1"}
    provider._inherited_native_sampling_identity = None
    provider.total_epochs = 2500
    provider.max_microbatch_cases = 8
    provider.effective_batch_size = 16
    provider.adapter = type("Adapter", (), {"SCHEMA": "wind_inherited_attention_adapter_v1"})()
    base_identity = {
        "coordinate_frame": {},
        "training": {},
        "environment_representation": {},
        "model_config": config,
        "adapter_state": {},
        "dataset_profile": "fixed24_v1",
        "subset_manifest_sha256": "a" * 64,
        "train_row_indices_sha256": "b" * 64,
        "normalizer_and_background_source": {"kind": "fixture"},
    }
    monkeypatch.setattr(WindContentHypergraphTask, "identity_payload", lambda self: dict(base_identity))

    identity = provider.identity_payload()
    serialization = provider.serialization_payload()

    assert identity["architecture"] == "IAG-Q"
    assert identity["representation"] == "iag_q"
    assert serialization["architecture"] == "IAG-Q"
    assert serialization["representation"] == "iag_q"


@pytest.mark.parametrize("sparse", [False, True])
def test_wind_q_standalone_loader_reconstructs_from_checkpoint_and_adapter_only(
    stub_q_loader, monkeypatch, tmp_path, sparse,
):
    payload = _checkpoint(sparse=sparse, tmp_path=tmp_path)
    monkeypatch.setattr(iag_loader, "load_trusted_checkpoint", lambda path, map_location: payload)
    loaded, adapter, saved = iag_loader.load_content_hypergraph_checkpoint(
        tmp_path / "in-memory-checkpoint.pt", device="cpu",
    )

    assert loaded.get_config()["architecture"] == "IAG-Q"
    assert loaded.get_config()["scene_executor"] == "cached_tokens"
    assert loaded.get_config()["query_executor"] == "cell_selected"
    assert adapter.state_payload()["schema"] == "wind_inherited_attention_adapter_v1"
    assert saved["arm"] == "IAG-Q"


@pytest.mark.parametrize("mutation", ["thermal_dataset", "provider_representation"])
def test_wind_q_loader_rejects_thermal_or_conflicting_representation(stub_q_loader, tmp_path, mutation):
    payload = _checkpoint(sparse=False, tmp_path=tmp_path)
    identity = payload["experiment_identity"]
    if mutation == "thermal_dataset":
        identity["recipe"]["dataset"] = "thermal"
        identity["recipe_sha256"] = _sha256(identity["recipe"])
    else:
        identity["provider_identity"]["representation"] = "iag_c"

    with pytest.raises(ValueError, match="unsupported dataset/representation|differs from its recipe"):
        iag_loader._load_inherited_attention_checkpoint(payload, device="cpu")


@pytest.mark.parametrize("mutation", ["source_architecture", "review_sha", "parent_review", "source_epoch"])
def test_wind_q_sparse_loader_rejects_a_non_q_or_unreviewed_e500_parent(
    stub_q_loader, tmp_path, mutation,
):
    payload = _checkpoint(sparse=True, tmp_path=tmp_path)
    contract = payload["branch_parent"]["inherited_attention_sparse_child"]
    if mutation == "source_architecture":
        contract["source_architecture"] = "IAG-C"
    elif mutation == "review_sha":
        contract["representation_review_sha256"] = "c" * 64
    elif mutation == "parent_review":
        payload["branch_parent"]["identity"]["representation_review"]["sha256"] = "c" * 64
    else:
        payload["branch_parent"]["epoch"] = 499

    with pytest.raises(ValueError, match="exact reviewed IAG-Q e500 parent branch"):
        iag_loader._load_inherited_attention_checkpoint(payload, device="cpu")
