from __future__ import annotations

import json
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import h5py

from channelthermal.local_surrogate.model import LocalModuleConfig, LocalModuleSurrogate
from channelthermal.resources import DatasetRegistry
from channelthermal.plugin import ThermalChannelPlugin
from honf_runtime.compat import load_trusted_checkpoint, strip_module_prefix
from honf_runtime.paths import PROJECT_ROOT
from honf_runtime.config_loader import load_config_bundle
from honf_runtime.run_store import RunStore
from honf_runtime.registry import load_object
from channelthermal.local_surrogate.spec import THERMAL_DISK_SPEC
from channelthermal.workflows.train_local import save_checkpoint
from channelthermal.workflows.evaluate_forward import latest_run_dir, resolve_checkpoint_arg
from channelthermal.environment import ChannelThermalEnvironmentBuilder
from honf_forward_core.decoder import rectangular_boundary_features


def test_named_datasets_match_manifest_contract() -> None:
    local_map = PROJECT_ROOT / "Case_ThermalChannel" / "Dataset" / "dataset_locations.local.json"
    if not local_map.exists():
        pytest.skip("external ThermalChannel datasets are not mapped on this machine")
    registry = DatasetRegistry(
        "project://Case_ThermalChannel/Dataset/dataset_manifest.json",
        local_map,
    )
    for dataset_id in ("thermal_disk_local_v1", "thermal_channel_global_v1"):
        resource = registry.resolve(dataset_id)
        registry.validate(resource)
        assert resource.path.is_file()
        assert len(resource.fingerprint) == 64


def test_historical_local_checkpoint_loads_strictly_and_is_finite() -> None:
    path = (
        PROJECT_ROOT.parent
        / "1_Demo_ChannelThermal"
        / "Saved_Model_LocalModule"
        / "Run_0003_20260507_224352"
        / "latest_model.pt"
    )
    if not path.exists():
        return
    checkpoint = load_trusted_checkpoint(path, map_location="cpu")
    config = LocalModuleConfig.from_dict(checkpoint.get("model_config", {}))
    model = LocalModuleSurrogate(config)
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    torch.manual_seed(123)
    params = torch.randn(2, config.module_param_dim)
    theta = torch.linspace(0.0, 2.0 * torch.pi, 17)[:-1]
    ports = torch.cat(
        [
            torch.stack([theta, torch.cos(theta), torch.sin(theta)], dim=-1).unsqueeze(0).expand(2, -1, -1),
            torch.randn(2, 16, config.port_token_dim - 3),
        ],
        dim=-1,
    )
    with torch.no_grad():
        output = model(params, ports, torch.rand(2, 7, 2) * 2.0 - 1.0)
    for value in output.values():
        assert torch.isfinite(value).all()


def test_current_local_checkpoint_contains_resume_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = LocalModuleConfig(
        module_param_dim=7,
        port_token_dim=5,
        interface_target_dim=2,
        hidden_dim=16,
        latent_dim=16,
        num_port_latents=4,
        num_heads=4,
        num_layers=1,
        coord_fourier_frequencies=2,
    )
    model = LocalModuleSurrogate(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    stats = {"internal_temperature_mean": np.asarray([1.0], dtype=np.float32)}
    dataset = SimpleNamespace(
        module_param_names=[f"p{i}" for i in range(7)],
        port_input_feature_names=[f"x{i}" for i in range(5)],
        interface_target_names=["surface_temperature", "normal_flux"],
        normalizer=SimpleNamespace(stats=stats),
    )
    path = tmp_path / "local.pt"
    save_checkpoint(
        path,
        model=model,
        model_config=config,
        train_config={"dataset": {}},
        dataset=dataset,
        epoch=3,
        best_metric=0.25,
        optimizer=optimizer,
    )
    checkpoint = load_trusted_checkpoint(path, map_location="cpu")
    assert checkpoint["optimizer_state_dict"] is not None
    assert checkpoint["epoch"] == 3
    assert set(checkpoint["rng_state"]) == {"python", "numpy", "torch", "cuda"}


def test_evaluation_manifest_inventories_artifacts(tmp_path) -> None:
    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")
    ThermalChannelPlugin._write_artifact_manifest(
        tmp_path,
        filename="evaluation_manifest.json",
        kind="eval_local",
        source_run=tmp_path.parent,
        checkpoint="best",
    )
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["artifacts"] == [{"path": "metrics.json", "size_bytes": 3}]


def test_case_validation_rejects_unknown_nested_setting() -> None:
    bundle = load_config_bundle("project://src/config_core/forward/enhanced_honf_pairwise.json")
    case = copy.deepcopy(bundle.case)
    case["model"]["physical_correction"]["refinment_steps"] = 1
    invalid = replace(bundle, case=case)
    with pytest.raises(ValueError, match="refinment_steps"):
        ThermalChannelPlugin().validate_config(invalid)


def test_evaluation_defaults_respect_workflow_surface() -> None:
    defaults = {
        "split": "test",
        "query_batch_size": 1024,
        "organization_view": "all",
        "return_routing_maps": True,
    }
    local_argv: list[str] = []
    ThermalChannelPlugin._append_evaluation_defaults(local_argv, defaults, workflow="local_module")
    assert local_argv == ["--split", "test"]
    compare_argv: list[str] = []
    ThermalChannelPlugin._append_evaluation_defaults(compare_argv, defaults, workflow="compare")
    assert "--query-batch-size" in compare_argv
    assert "--organization-view" not in compare_argv


def test_local_module_spec_references_real_implementations() -> None:
    assert THERMAL_DISK_SPEC.module_id == "thermal_disk"
    assert THERMAL_DISK_SPEC.embedded_in_parent_checkpoint
    for dotted in (
        THERMAL_DISK_SPEC.model_factory,
        THERMAL_DISK_SPEC.checkpoint_loader,
        THERMAL_DISK_SPEC.train_workflow,
        THERMAL_DISK_SPEC.evaluate_workflow,
        THERMAL_DISK_SPEC.coupling_adapter,
    ):
        assert load_object(dotted) is not None


def test_data_root_uses_manifest_relative_path(tmp_path, monkeypatch) -> None:
    data_file = tmp_path / "family" / "packed.h5"
    data_file.parent.mkdir()
    with h5py.File(data_file, "w") as handle:
        handle.create_dataset("required", data=[1])
    manifest = {
        "schema_version": 1,
        "datasets": {
            "fixture": {
                "filename": "packed.h5",
                "relative_path": "family/packed.h5",
                "size_bytes": data_file.stat().st_size,
                "required_keys": ["required"],
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("HONF_DATA_ROOT", str(tmp_path))
    registry = DatasetRegistry(manifest_path, None)
    resource = registry.resolve("fixture")
    assert resource.path == data_file.resolve()
    registry.validate(resource)


def test_case_query_features_match_legacy_checkpoint_boundary_order() -> None:
    query_xy = torch.tensor([[[0.0, 0.0], [3.0, 2.0], [12.0, 4.0]]])
    case_features = ChannelThermalEnvironmentBuilder.query_features(
        query_xy,
        domain_length_x=12.0,
        domain_length_y=4.0,
    )
    legacy_features = rectangular_boundary_features(query_xy, 12.0, 4.0)
    assert ChannelThermalEnvironmentBuilder.query_feature_names == (
        "x_norm",
        "y_norm",
        "bottom_wall_distance_norm",
        "top_wall_distance_norm",
        "inlet_distance_norm",
        "outlet_distance_norm",
    )
    assert torch.equal(case_features, legacy_features)


def test_forward_checkpoint_resolution_requires_explicit_fallback(tmp_path) -> None:
    run = tmp_path / "Run_0001_20260101_000000_fixture"
    run.mkdir()
    (run / "best_model.pt").write_bytes(b"best")
    args = SimpleNamespace(
        checkpoint="best_predicted",
        run_id="0001",
        saved_root=str(tmp_path),
        allow_checkpoint_fallback=False,
    )
    assert resolve_checkpoint_arg(args).name == "best_predicted_model.pt"
    args.allow_checkpoint_fallback = True
    assert resolve_checkpoint_arg(args) == (run / "best_model.pt").resolve()


def test_forward_run_lookup_rejects_ambiguous_ids(tmp_path) -> None:
    (tmp_path / "Run_0001_20260101_000000_a").mkdir()
    (tmp_path / "Run_0001_20260102_000000_b").mkdir()
    with pytest.raises(RuntimeError, match="ambiguous"):
        latest_run_dir(tmp_path, "0001")


def test_evaluation_uses_immutable_source_run_config(tmp_path) -> None:
    bundle = load_config_bundle("project://src/config_core/forward/enhanced_honf_pairwise.json")
    store = RunStore(tmp_path)
    proposal = store.propose(
        case_id="ThermalChannel",
        workflow="forward",
        model_family="honf_forward",
        run_id="0044",
        run_name="evaluation_source",
    )
    run_dir = store.create(proposal, bundle)
    saved = json.loads((run_dir / "configs" / "resolved_config.json").read_text())
    effective = ThermalChannelPlugin()._evaluation_effective_config(
        bundle,
        run_dir.parent,
        "0044",
        workflow="forward",
        checkpoint="best",
    )
    assert effective == saved
