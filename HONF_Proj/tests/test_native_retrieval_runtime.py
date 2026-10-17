"""Data-free reconstruction tests for strict Round7 child checkpoints."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_forward_core.native_retrieval.core import NativeRetrievalCore
from honf_runtime import compat
from honf_runtime import native_retrieval_protocol as protocol
from honf_runtime import native_retrieval_runtime as runtime
from honf_runtime.unified_training import SamplingKey, _sampling_dataset_id


def _parent_components(dataset):
    torch.manual_seed(19)
    if dataset == "thermal":
        from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2

        core = ContentHypergraphCoreV2(
            module_input_dim=8, environment_input_dim=8, coordinate_dim=2,
            query_input_dim=0, operating_input_dim=14, output_dim=3,
            architecture="A2-D", context_dim=32, num_heads=4, context_depth=1,
            num_hyperedges=6, thermal_response_rank=4, local_hidden_dim=32,
            dropout=0.0, init_seed=19,
        )
        stats = {
            "field_mean_by_channel": np.zeros(5, dtype=np.float32),
            "field_std_by_channel": np.ones(5, dtype=np.float32),
            "interface_targets_mean": np.zeros(2, dtype=np.float32),
            "interface_targets_std": np.ones(2, dtype=np.float32),
            "internal_temperature_std": np.ones(1, dtype=np.float32),
        }
        adapter = ThermalContentHypergraphAdapterV2(stats, core=core, nx=8, ny=6)
        environment = {
            "token_shape_ny_nx": [8, 24], "count": 192, "geometry_only": True,
            "heating_in_scene": False, "temperature_or_flow_target_features": False,
        }
        return core, adapter, environment

    from windfarm.content_hypergraph import WindContentHypergraphAdapterV3
    from windfarm.normalization import VelocityNormalizer, VerticalProfileBaseline

    core = ContentHypergraphCoreV2(
        module_input_dim=2, environment_input_dim=7, coordinate_dim=3,
        query_input_dim=7, operating_input_dim=11, output_dim=3,
        architecture="A2-D", context_dim=32, num_heads=4, context_depth=1,
        num_hyperedges=6, thermal_response_rank=0, local_hidden_dim=32,
        dropout=0.0, init_seed=19,
    )
    normalizer = VelocityNormalizer(np.zeros(3), np.ones(3), np.ones(3))
    profile = VerticalProfileBaseline(
        np.asarray([0.0, 2.0]), np.ones((2, 3)), np.ones(2, dtype=np.int64),
    )
    adapter = WindContentHypergraphAdapterV3(normalizer, profile)
    environment = {
        "token_shape": [4, 4, 4], "record_count": 64,
        "measure": "geometry-only native domain-support quadrature in rotor_diameters^3",
        "field_sensors": False,
    }
    return core, adapter, environment


def _round7_payload(dataset, treatment, tmp_path, monkeypatch):
    # Recipe files are part of the maintained Round3 source contract.
    from pathlib import Path

    recipe_path = Path(protocol.__file__).resolve().parents[2] / "src/config_core/forward/content_hypergraph" / (
        f"{dataset}_a2-d_source_read_development_d0_v1.json"
    )
    base_recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    source_checkpoint = tmp_path / f"{dataset}_d0_e500.pt"
    source_checkpoint.write_bytes(b"synthetic memory-only parent sentinel")
    parent_core, parent_adapter, environment = _parent_components(dataset)
    subset = "a" * 64

    def provider_for(core, adapter):
        return {
            "dataset": dataset,
            "subset_manifest_sha256": subset,
            "model_config": adapter.get_config() if dataset == "thermal" else core.get_config(),
            "adapter_state": adapter.state_payload(),
            "environment_representation": copy.deepcopy(environment),
        }

    source_provider = provider_for(parent_core, parent_adapter)
    source_identity = {
        "recipe": base_recipe,
        "recipe_sha256": protocol.canonical_sha256(base_recipe),
        "provider_identity": source_provider,
    }
    fake_payload = {
        "epoch": 500, "arm": "A2-D", "experiment_identity": source_identity,
    }
    monkeypatch.setattr(
        protocol, "_source_contract",
        lambda _path: (copy.deepcopy(fake_payload), copy.deepcopy(source_identity), "c" * 64),
    )
    recipe = protocol.make_native_retrieval_recipe(
        source_checkpoint,
        treatment=treatment,
        native_backend="full_direct" if treatment == "D-control" else "selected",
        tau_q=0.0 if treatment == "D-control" else 0.20,
    )

    if treatment == "G-native-M":
        native = NativeRetrievalCore.from_a2d(
            parent_core, native_tau_mm=recipe["native_tau_mm"],
            native_tau_me=recipe["native_tau_me"], native_tau_q=recipe["native_tau_q"],
            native_backend=recipe["native_backend"], environment_value_reader=False,
        )
        native.set_native_training_epoch(501)
        if dataset == "thermal":
            from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2

            model = ThermalContentHypergraphAdapterV2(
                parent_adapter.normalization_stats, core=native,
                nx=parent_adapter.nx, ny=parent_adapter.ny,
                environment_nx=parent_adapter.environment_nx,
                environment_ny=parent_adapter.environment_ny,
            )
            adapter = model
        else:
            model = native
            adapter = parent_adapter
        child_provider = provider_for(native, adapter)
        # The provider builder records the same geometry-only metadata in each arm.
        child_provider["environment_representation"] = copy.deepcopy(environment)
    else:
        model = parent_adapter if dataset == "thermal" else parent_core
        adapter = model if dataset == "thermal" else parent_adapter
        child_provider = source_provider

    dataset_id = _sampling_dataset_id(source_provider)
    declaration = protocol.native_retrieval_child_declaration(recipe)
    parent_identity = {
        "recipe": base_recipe,
        "recipe_sha256": protocol.canonical_sha256(base_recipe),
        "provider_identity": source_provider,
    }
    routing_state = runtime._expected_history_routing_state(recipe, 501)
    cadence = {
        "absolute_start_epoch": 501,
        "train_case_visits_per_epoch": 1,
        "optimizer_updates_per_epoch": 1,
        "case_counts_per_update": [1],
        "microbatch_cases": 1,
        "effective_cases": 1,
        "primary_queries_per_case": 1,
        "response_cadence": "not_applicable" if dataset == "wind" else "synthetic fixture",
        "execution": copy.deepcopy(recipe["execution"]),
    }
    identity = {
        "family": protocol.NATIVE_RETRIEVAL_FAMILY,
        "model_family": protocol.NATIVE_RETRIEVAL_FAMILY,
        "workflow": protocol.NATIVE_RETRIEVAL_WORKFLOW,
        "protocol": recipe["protocol"],
        "treatment": treatment,
        "recipe": recipe,
        "recipe_sha256": protocol.canonical_sha256(recipe),
        "provider_identity": child_provider,
        "sampling_dataset_id": dataset_id,
        "sampling_identity": {
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "source_sampling_dataset_id": dataset_id,
            "matches_d0_e500_namespace": True,
            "arm": "A2-D",
            "fit_epoch_is_absolute": True,
            "sampler_epoch_mapping": "identity",
        },
        "engine_config": {
            "sampling_version": SamplingKey.CASE_EPOCH_VERSION,
            "deterministic_case_order": True,
        },
        "training_cadence": cadence,
    }
    payload = {
        "workflow": protocol.NATIVE_RETRIEVAL_WORKFLOW,
        "experiment_identity": identity,
        "arm": "A2-D",
        "epoch": 501,
        "model_state_dict": model.state_dict(),
        "branch_parent": {
            "path": recipe["source_checkpoint"],
            "sha256": recipe["source_checkpoint_sha256"],
            "epoch": 500,
            "identity": parent_identity,
            "native_retrieval_child": declaration,
        },
        "sampler_state": {
            "completed_epoch": 501,
            "next_epoch": 502,
            "absolute_sampling_epoch": 501,
        },
        "history": [{
            "epoch": 501, "case_visits": 1, "optimizer_updates": 1,
            "microbatches": 1, "partial_update_cases": 1,
            "native_routing_state": routing_state,
        }],
    }
    return payload, model, adapter, child_provider


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
@pytest.mark.parametrize("treatment", ["D-control", "G-native-M"])
def test_native_checkpoint_rebuilds_core_and_adapter_without_dataset_access(
    dataset, treatment, tmp_path, monkeypatch,
):
    payload, expected_model, expected_adapter, _ = _round7_payload(
        dataset, treatment, tmp_path, monkeypatch,
    )
    monkeypatch.setattr(compat, "load_trusted_checkpoint", lambda *_args, **_kwargs: copy.deepcopy(payload))
    model, adapter, saved = runtime.load_native_retrieval_checkpoint("/unused/round7.pt", device="cpu")
    assert saved["epoch"] == 501
    assert model.get_config() == expected_model.get_config()
    if dataset == "thermal":
        assert adapter is model
        assert adapter.state_payload() == expected_adapter.state_payload()
    else:
        assert adapter.state_payload() == expected_adapter.state_payload()
    for name, value in expected_model.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
