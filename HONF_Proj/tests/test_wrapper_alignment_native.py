"""Opt-in retained checkpoint replay and disposable native wrapper updates.

Run with HONF_ALIGNMENT_THERMAL_CHECKPOINT, HONF_ALIGNMENT_WIND_CHECKPOINT,
HONF_ALIGNMENT_WIND_DATASET and CUDA_VISIBLE_DEVICES=1 (or use CPU).
No checkpoint or training output is written. Historical methods are loaded
from the declared git ref; unchanged fine-kernel modules remain shared.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from honf_runtime.compat import recursive_to_device

ROOT = Path(__file__).resolve().parents[2]


def _historical_class(path: str, module_name: str, class_name: str):
    ref = os.environ.get("HONF_ALIGNMENT_BASE_REF", "0076b2898d47b1d12a87e89ff10f5005b12543b5")
    source = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    module = types.ModuleType(module_name)
    module.__package__ = module_name.rsplit(".", 1)[0]
    sys.modules[module_name] = module
    exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)  # noqa: S102 - declared trusted repository revision
    return getattr(module, class_name)


def _retained_methods(model, case):
    historical = copy.deepcopy(model)
    historical.__class__ = _historical_class(f"HONF_Proj/Case_{case}/src/{'channelthermal' if case == 'ThermalChannel' else 'windfarm'}/model.py",
                                            f"{'channelthermal' if case == 'ThermalChannel' else 'windfarm'}._alignment_retained_model",
                                            "ChannelThermalHONFModel" if case == "ThermalChannel" else "WindFarmForwardModel")
    historical.core.__class__ = _historical_class("HONF_Proj/src/honf_forward_core/interface_fields/core.py",
                                                 "honf_forward_core.interface_fields._alignment_retained_core", "InterfaceFieldCore")
    return historical


def _device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_canonical_common_latent_reads_through_both_case_core_facades():
    from channelthermal.config import ChannelThermalHONFConfig
    from channelthermal.model import ChannelThermalHONFModel
    from windfarm.model import WindFarmForwardModel

    from honf_forward_core.config import BatchData, UnifiedForwardConfig

    payload = {"forward_architecture": "three_term_full_access_honf", "spatial_dim": 2,
               "field_dim": 5, "hidden_dim": 16, "coordinate_scale": [12., 6.],
               "interface_model": {"message_hidden_dim": 8, "attention_heads": 2,
                                   "receiver_chunk_size": 3, "activation_checkpointing": False}}
    thermal = ChannelThermalHONFModel(ChannelThermalHONFConfig.from_dict({"core_honf": payload,
                                                                       "channelthermal": {"use_local_surrogate": False}}))
    wind = WindFarmForwardModel(UnifiedForwardConfig.from_dict(payload))
    generator = torch.Generator().manual_seed(5)
    batch = BatchData(module_centers=torch.rand(1, 3, 2, generator=generator), module_present=torch.ones(1, 3),
                      module_features=torch.rand(1, 3, 4, generator=generator), global_context=torch.rand(1, 4, generator=generator),
                      query_xy=torch.rand(1, 7, 2, generator=generator), query_time=None, target_field=None,
                      case_name="canonical", metadata={}, env_coords=torch.rand(1, 5, 2, generator=generator),
                      env_features=torch.rand(1, 5, 3, generator=generator), env_weights=torch.ones(1, 5))
    for core in (thermal.core, wind.core):
        encoded = core.encode_case(batch)
        prepared = core.prepare(encoded, encoded.module_tokens)
        core.read(prepared, batch.query_xy)
    wind.core.load_state_dict(thermal.core.state_dict(), strict=True)
    results = []
    for core in (thermal.core, wind.core):
        encoded = core.encode_case(batch)
        prepared = core.prepare(encoded, encoded.module_tokens)
        results.append((encoded.module_tokens, core.read(prepared, batch.query_xy).context))
    for actual, expected in zip(results[0], results[1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_thermal_environment_physical_length_is_independent_of_normalized_weights():
    from channelthermal.environment import ChannelThermalEnvironmentBuilder

    from honf_forward_core.config import BatchData, UnifiedForwardConfig
    from honf_forward_core.interface_fields.core import InterfaceFieldCore

    environment = ChannelThermalEnvironmentBuilder()(batch_size=1, num_env_tokens_x=24, num_env_tokens_y=8,
                                                     domain_length_x=12., domain_length_y=6., device=torch.device("cpu"),
                                                     dtype=torch.float32)
    expected_length = (12. * 6. / 192) ** .5
    torch.testing.assert_close(environment.env_characteristic_lengths, torch.full((1, 192), expected_length))
    normalized = environment.env_weights / environment.env_weights.sum(1, keepdim=True)
    assert not torch.allclose(environment.env_characteristic_lengths, normalized.sqrt())
    core = InterfaceFieldCore(UnifiedForwardConfig.from_dict({"forward_architecture": "three_term_full_access_honf",
                                                              "hidden_dim": 16, "field_dim": 5,
                                                              "interface_model": {"message_hidden_dim": 8, "attention_heads": 2}}))
    batch = BatchData(module_centers=torch.zeros(1, 1, 2), module_present=torch.ones(1, 1),
                      module_features=torch.ones(1, 1, 3), global_context=torch.ones(1, 4),
                      query_xy=torch.zeros(1, 1, 2), query_time=None, target_field=None, case_name="physical-length",
                      metadata={}, env_coords=environment.env_coords, env_features=environment.env_features,
                      env_weights=normalized, env_characteristic_lengths=environment.env_characteristic_lengths)
    encoded = core.encode_case(batch)
    torch.testing.assert_close(encoded.env_weights, normalized)
    torch.testing.assert_close(encoded.env_characteristic_lengths, environment.env_characteristic_lengths)


@pytest.mark.skipif(not os.environ.get("HONF_ALIGNMENT_THERMAL_CHECKPOINT"), reason="Needs local retained Thermal checkpoint")
def test_native_thermal_replay_chunks_and_disposable_optimizer_step():
    from channelthermal.data.collation import ChannelThermalBatchCollator
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.evaluation.loading import load_model
    from channelthermal.training.epoch import make_model_inputs, run_epoch
    from channelthermal.training.optimizer import build_forward_optimizer

    device = _device()
    model, checkpoint = load_model(Path(os.environ["HONF_ALIGNMENT_THERMAL_CHECKPOINT"]), device)
    settings = checkpoint["train_config"]["dataset"]
    dataset = GlobalChannelThermalDataset(settings["packed_h5_path"], split="train", points_per_case=17,
                                         normalize_inputs=settings["normalize_inputs"], normalize_targets=settings["normalize_targets"],
                                         random_point_sampling=False, seed=0, include_grid=False, include_structure_targets=False)
    loader = DataLoader(dataset, batch_size=1, collate_fn=ChannelThermalBatchCollator())
    batch = recursive_to_device(next(iter(loader)), device)
    inputs = make_model_inputs(batch, local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
                               return_predicted_port_outputs=False, return_port_global_consistency=True)
    retained = _retained_methods(model, "ThermalChannel")
    with torch.no_grad():
        original = retained(**inputs)
        actual = model(**inputs, return_prepared_state=True)
        for key in ("pred_field", "pred_interface", "pred_internal_temperature", "used_port_tokens"):
            if key in original:
                torch.testing.assert_close(actual[key], original[key], rtol=2e-5, atol=2e-5)
        prepared = actual["prepared_state"]
        chunks = [model.decode_prepared(prepared, batch["query_xy"][:, i:i+3])["pred_field"] for i in range(0, 17, 3)]
        torch.testing.assert_close(torch.cat(chunks, 1), actual["pred_field"], rtol=2e-5, atol=2e-5)
    del retained
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    before = model.core.common.field_head.net[-1].weight.detach().clone()
    optimizer, _ = build_forward_optimizer(model, checkpoint["train_config"]["training"])
    model.set_training_progress(epoch=1, total_epochs=5000)
    metrics = run_epoch(model, loader, device, checkpoint["train_config"]["loss"], optimizer=optimizer, scaler=None,
                        amp=False, max_batches=1, local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
                        effective_internal_temperature_weight=1., effective_interface_weight=.2,
                        predicted_consistency_weight=.0005, gradient_clip_norm=1.)
    assert metrics["epoch_optimizer_steps"] == 1
    assert np.isfinite(metrics["loss_total"])
    assert not torch.equal(before, model.core.common.field_head.net[-1].weight)
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name]), name
    print({"Thermal_epoch": checkpoint["epoch"], "case": batch["case_id"], "native_step": metrics["loss_total"]})


@pytest.mark.skipif(not (os.environ.get("HONF_ALIGNMENT_WIND_CHECKPOINT") and os.environ.get("HONF_ALIGNMENT_WIND_DATASET")), reason="Needs local retained Wind checkpoint and dataset")
def test_native_wind_replay_chunks_and_disposable_optimizer_step():
    from windfarm.data import WindFarmNativeView, case_batch
    from windfarm.workflows.evaluate_forward import load_checkpoint

    device = _device()
    view = WindFarmNativeView(os.environ["HONF_ALIGNMENT_WIND_DATASET"])
    case = view.run(0)
    fractions = np.array([[.1, .3, .2], [.3, .4, .3], [.5, .5, .5], [.8, .6, .7]], dtype=np.float32)
    points = case.support.lower_D + fractions * (case.support.upper_D - case.support.lower_D)
    batch = case_batch(case, points).to(device)
    model, checkpoint = load_checkpoint(os.environ["HONF_ALIGNMENT_WIND_CHECKPOINT"], device=device, materialization_batch=batch)
    retained = _retained_methods(model, "WindFarm")
    with torch.no_grad():
        actual = model(batch)
        original = retained(batch)
        torch.testing.assert_close(actual["pred_field"], original["pred_field"], rtol=2e-5, atol=2e-5)
        prepared = model.prepare_case(batch)
        pieces = [model.predict_standardized(prepared, batch.query_xy[:, i:i+1], batch.query_features[:, i:i+1]) for i in range(4)]
        torch.testing.assert_close(torch.cat(pieces, 1), actual["pred_field"], rtol=2e-5, atol=2e-5)
    del retained
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    before = model.core.common.field_head.net[-1].weight.detach().clone()
    model.train()
    loss = model(batch)["pred_field"].square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.core.common.field_head.net[-1].weight)
    print({"Wind_epoch": checkpoint["epoch"], "case": case.case, "native_step": float(loss.detach())})


@pytest.mark.skipif(not os.environ.get("HONF_ALIGNMENT_WIND_DATASET"), reason="Needs local native Wind geometry")
@pytest.mark.parametrize("architecture", ["adaptive_receiver_hypergraph_honf", "overlap_control_hypergraph_honf", "local_overlap_hypergraph_honf"])
def test_disposable_finalist_wind_three_dimensional_wrapper_step(architecture):
    from windfarm.data import WindFarmNativeView, case_batch
    from windfarm.model import WindFarmForwardModel

    from honf_forward_core.config import UnifiedForwardConfig
    from honf_forward_core.training.hypergraph_shadow import hard_value_soft_hypergraph_forward

    case = WindFarmNativeView(os.environ["HONF_ALIGNMENT_WIND_DATASET"]).run(0)
    points = case.support.lower_D + np.array([[.1, .3, .2], [.3, .4, .3], [.5, .5, .5], [.8, .6, .7]]) * case.support.extent_D
    batch = case_batch(case, points).to(_device())
    config = UnifiedForwardConfig.from_dict({"forward_architecture": architecture, "spatial_dim": 3, "field_dim": 3,
                                             "boundary_feature_mode": "none", "geometry_mode": "nonperiodic",
                                             "hidden_dim": 16, "coordinate_scale": [50., 38., 6.25],
                                             "interface_model": {"message_hidden_dim": 8, "attention_heads": 2,
                                                                 "receiver_chunk_size": 2, "activation_checkpointing": True}})
    model = WindFarmForwardModel(config).to(_device())
    with torch.no_grad():
        model.eval()
        model(batch)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    before = model.core.common.field_head.net[-1].weight.detach().clone()
    output = hard_value_soft_hypergraph_forward(model, batch)
    loss = output["pred_field"].square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.core.common.field_head.net[-1].weight)
    model.eval()
    with torch.no_grad():
        prepared = model.prepare_case(batch)
        exported = model.export_typed_hypergraph(prepared)
        assert exported["source_coords"]["E"].shape[-1] == 3
        torch.testing.assert_close(exported["source_lengths"]["E"], batch.env_characteristic_lengths)
    print({"Wind_compatibility_architecture": architecture, "native_case": case.case, "loss": float(loss.detach())})


@pytest.mark.skipif(not os.environ.get("HONF_ALIGNMENT_THERMAL_CHECKPOINT"), reason="Needs native local Thermal resources")
@pytest.mark.parametrize("arm", ["h-tree", "h-overlap", "h-local"])
def test_fresh_thermal_matched_initialization_and_native_shadow_step(arm):
    from channelthermal.data.collation import ChannelThermalBatchCollator
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.model import ChannelThermalHONFModel
    from channelthermal.plugin import create_plugin
    from channelthermal.training.campaign import copy_matched_physical_initial_state
    from channelthermal.training.epoch import make_model_inputs
    from channelthermal.workflows.train_forward import build_model_config

    from honf_forward_core.training.hypergraph_shadow import hard_value_soft_hypergraph_forward
    from honf_runtime.case_protocol import WorkflowRequest
    from honf_runtime.config_loader import load_config_bundle

    torch.manual_seed(0)
    bundle = load_config_bundle(f"src/config_core/forward/thermal_campaign/{arm}_e100.json")
    cfg = create_plugin()._forward_config(bundle, WorkflowRequest("forward"), ROOT / "HONF_Proj/diagnostics/generated/disposable")
    settings = cfg["dataset"]
    dataset = GlobalChannelThermalDataset(settings["packed_h5_path"], split="train", points_per_case=17,
                                         normalize_inputs=True, normalize_targets=True, random_point_sampling=False,
                                         include_grid=False, include_structure_targets=False)
    device = _device()
    batch = recursive_to_device(next(iter(DataLoader(dataset, batch_size=1, collate_fn=ChannelThermalBatchCollator()))), device)
    inputs = make_model_inputs(batch, local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
                               return_predicted_port_outputs=False, return_port_global_consistency=False)
    config = build_model_config(cfg, dataset)
    model = ChannelThermalHONFModel(config).to(device)
    model.set_global_target_normalization(dataset.normalizer.stats, normalize_targets=True)
    model.eval()
    with torch.no_grad():
        model(**inputs)
    canonical_config = copy.deepcopy(config)
    canonical_config.core_honf.forward_architecture = "three_term_full_access_honf"
    torch.manual_seed(0)
    canonical = ChannelThermalHONFModel(canonical_config).to(device)
    canonical.eval()
    canonical.set_global_target_normalization(dataset.normalizer.stats, normalize_targets=True)
    with torch.no_grad():
        canonical(**inputs)
    inventory = copy_matched_physical_initial_state(model, canonical)
    assert inventory["loaded"]
    del canonical
    model.train()
    model.set_training_progress(epoch=1, total_epochs=5000)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=3e-4)
    output = hard_value_soft_hypergraph_forward(model, **inputs)
    loss = (output["pred_field"] - batch["field_targets"]).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    organizer_grad = sum(float(parameter.grad.abs().sum()) for name, parameter in model.named_parameters()
                         if name.startswith("core.backend.organizer.") and parameter.grad is not None)
    assert organizer_grad > 0
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name])
    model.eval()
    with torch.no_grad():
        result = model(**inputs, return_prepared_state=True)
        exported = model.export_typed_hypergraph(result["prepared_state"])
        assert exported["source_types"] == ("M", "E")
    print({"fresh_arm": arm, "case": batch["case_id"], "physical_tensors_matched": len(inventory["loaded"]),
           "loss": float(loss.detach()), "organizer_gradient_l1": organizer_grad})
    if arm == "h-overlap":
        from channelthermal.training.campaign_response import NativeCampaignResponse

        model.train()
        model.set_training_progress(epoch=101, total_epochs=5000)
        optimizer.zero_grad(set_to_none=True)
        native = hard_value_soft_hypergraph_forward(model, **inputs)
        value_loss = (native["pred_field"] - batch["field_targets"]).square().mean()
        response = NativeCampaignResponse(model, dataset, cfg["dataset"], cfg["training"]["campaign"])
        auxiliary, measured = response(101, value_loss, 1.)
        assert torch.isfinite(auxiliary)
        assert measured["response_examples"] == 2
        (value_loss + auxiliary).backward()
        optimizer.step()
        print({"response_native_integration": measured})
