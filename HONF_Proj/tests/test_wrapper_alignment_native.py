"""Opt-in retained checkpoint replay and disposable native wrapper updates.

Run with HONF_ALIGNMENT_THERMAL_CHECKPOINT, HONF_ALIGNMENT_WIND_CHECKPOINT,
HONF_ALIGNMENT_WIND_DATASET and CUDA_VISIBLE_DEVICES=1 (or use CPU).
No checkpoint or training output is written. Historical methods are loaded
from the declared git ref; unchanged fine-kernel modules remain shared.
"""

from __future__ import annotations

import copy
import json
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
        for key in ("pred_field", "pred_interface", "pred_internal_temperature", "pred_port_condition",
                    "pred_port_condition_raw", "local_port_condition_used"):
            assert key in original, f"Missing retained output: {key}"
            assert key in actual, f"Missing current output: {key}"
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
                        predicted_consistency_weight=.0005, gradient_clip_norm=1.,
                        campaign_config={"require_full_epoch": False})
    assert metrics["epoch_optimizer_steps"] == 1
    assert model.campaign_last_forward_work["native_training"]["hard"]["unlabelled"]["prepare_calls"] == 3
    assert not model.campaign_last_forward_work["executor_ledger_available"]
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
@pytest.mark.parametrize("hidden_dim", [16, 256])
def test_disposable_finalist_wind_three_dimensional_wrapper_step(architecture, hidden_dim):
    """Native Q4 construction/AD at tiny and campaign widths, without pretrained case-head transfer."""
    from channelthermal.training.campaign_work import CampaignForwardWork
    from windfarm.data import WindFarmNativeView, case_batch
    from windfarm.model import WindFarmForwardModel
    from windfarm.study_spatial import native_coordinates

    from honf_forward_core.config import UnifiedForwardConfig
    from honf_forward_core.training.hypergraph_shadow import hard_value_soft_hypergraph_forward

    torch.manual_seed(32)
    case = WindFarmNativeView(os.environ["HONF_ALIGNMENT_WIND_DATASET"]).run(0)
    flat = np.arange(1, 5, dtype=np.int64) * (int(np.prod(case.run.shape_nxyz)) // 5)
    points = native_coordinates(case.run, flat, case.diameter_m)
    batch = case_batch(case, points, velocity_mps=np.asarray(case.run.U[flat], dtype=np.float32)).to(_device())
    payload = {"forward_architecture": architecture, "spatial_dim": 3, "field_dim": 3,
               "boundary_feature_mode": "none", "geometry_mode": "nonperiodic",
               "hidden_dim": hidden_dim, "coordinate_scale": [50., 38., 6.25],
               "interface_model": {"message_hidden_dim": 8, "attention_heads": 2,
                                   "receiver_chunk_size": 2, "activation_checkpointing": True}}
    profile_path = None
    if hidden_dim == 256:
        arm = {"adaptive_receiver_hypergraph_honf": "h-tree", "overlap_control_hypergraph_honf": "h-overlap",
               "local_overlap_hypergraph_honf": "h-local"}[architecture]
        profile_path = ROOT / f"HONF_Proj/src/config_core/forward/thermal_campaign/{arm}_e100.json"
        profile = json.loads(profile_path.read_text())["model"]["core_honf"]
        for key in ("hidden_dim", "dropout", "use_layer_norm", "query_time_mode", "query_fourier_frequencies",
                    "position_fourier_frequencies", "use_position_fourier_for_modules", "use_position_fourier_for_env"):
            payload[key] = profile[key]
        payload["interface_model"] = copy.deepcopy(profile["interface_model"])
    config = UnifiedForwardConfig.from_dict(payload)
    model = WindFarmForwardModel(config).to(_device())
    with torch.no_grad():
        model.eval()
        model(batch)
    model.train()
    model.set_training_progress(epoch=1, total_epochs=5000)
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    physical_buffers = {name: value.detach().clone() for name, value in model.named_buffers()
                        if not name.startswith("core.backend.organizer.")}
    physical = [(name, value) for name, value in model.named_parameters()
                if value.requires_grad and not name.startswith("core.backend.organizer.")]
    # Match stochastic admission draws for the reference and wrapped hard call.
    devices = [batch.query_xy.device.index] if batch.query_xy.is_cuda else []
    counter = getattr(model.core.backend.organizer, "exercise_counter", None)
    saved_counter = counter.detach().clone() if counter is not None else None
    try:
        with torch.random.fork_rng(devices=devices):
            hard = model(batch)["pred_field"]
            hard_loss = (hard - batch.target_field).square().mean()
            hard_gradients = torch.autograd.grad(hard_loss, [value for _, value in physical], allow_unused=True)
    finally:
        if counter is not None:
            counter.copy_(saved_counter)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    before = model.core.common.field_head.net[-1].weight.detach().clone()
    with CampaignForwardWork(model.core) as measured_work:
        output = hard_value_soft_hypergraph_forward(model, batch)
    work_snapshot = copy.deepcopy(measured_work.records)
    assert set(work_snapshot) == {"hard", "soft"}
    for mode in work_snapshot.values():
        assert set(mode) == {"P0"}  # Wind has one native phase, without thermal port refinement.
        phase = mode["P0"]
        assert phase["prepare_calls"] == phase["read_calls"] == 1
        assert all(phase["ledgers"][f"hypergraph_{tau}_executed_rows"] > 0
                   for tau in ("MM", "ME", "EM", "QM", "QE"))
    torch.testing.assert_close(output["pred_field"], hard, rtol=0, atol=0)
    loss = (output["pred_field"] - batch.target_field).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert measured_work.records == work_snapshot  # Excludes activation-checkpoint backward recomputation.
    physical_gradient_max_error = 0.
    for (name, value), expected in zip(physical, hard_gradients):
        if expected is None:
            assert value.grad is None, name
        else:
            assert value.grad is not None and torch.isfinite(value.grad).all(), name
            torch.testing.assert_close(value.grad, expected, rtol=2e-4, atol=2e-5)
            physical_gradient_max_error = max(physical_gradient_max_error, float((value.grad - expected).abs().max()))
    organizer_gradient = sum(float(value.grad.abs().sum()) for name, value in model.named_parameters()
                             if name.startswith("core.backend.organizer.") and value.grad is not None)
    assert np.isfinite(organizer_gradient) and organizer_gradient > 0
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.core.common.field_head.net[-1].weight)
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name]), name
    for name, value in model.named_buffers():
        if name in physical_buffers:
            assert torch.equal(value, physical_buffers[name]), name
    model.eval()
    with torch.no_grad():
        prepared = model.prepare_case(batch)
        exported = model.export_typed_hypergraph(prepared)
        assert exported["source_coords"]["E"].shape[-1] == 3
        torch.testing.assert_close(exported["source_lengths"]["E"], batch.env_characteristic_lengths)
    compact_work = {mode: {phase: {"prepare_calls": value["prepare_calls"], "read_calls": value["read_calls"],
                                  **{count: sum(item for key, item in value["ledgers"].items() if key.endswith("_" + count))
                                     for count in ("executed_rows", "fine_calls", "skipped_eligible_pairs")}}
                          for phase, value in phases.items()} for mode, phases in work_snapshot.items()}
    print({"Wind_compatibility_architecture": architecture, "native_case": case.case,
           "native_grid_indices": flat.tolist(), "M": int(batch.module_present.sum()), "Q": 4, "hidden_dim": hidden_dim,
           "profile": str(profile_path) if profile_path is not None else None,
           "message_hidden_dim": config.interface_model.message_hidden_dim,
           "attention_heads": config.interface_model.attention_heads,
           "relative_fourier_frequencies": config.interface_model.relative_fourier_frequencies,
           "organizer_dim": getattr(model.core.backend.organizer, "node_encoder", [None])[0].out_features
               if architecture == "adaptive_receiver_hypergraph_honf" else None,
           "group_count": getattr(model.core.backend.organizer, "group_count", None),
           "control_dim": model.core.backend.organizer.control_dim,
           "loss": float(loss.detach()), "physical_parameters_with_gradient": sum(value is not None for value in hard_gradients),
           "physical_gradient_max_error": physical_gradient_max_error, "organizer_gradient_l1": organizer_gradient,
           "frozen_parameters": len(frozen), "physical_buffers_unchanged": len(physical_buffers),
           "forward_work": compact_work, "device": str(batch.query_xy.device),
           "scope": "disposable fresh native architecture/Q4 construction/AD step, no pretrained Thermal-to-Wind transfer; forward work excludes backward recomputation and export"})


@pytest.mark.skipif(not os.environ.get("HONF_ALIGNMENT_THERMAL_CHECKPOINT"), reason="Needs native local Thermal resources")
@pytest.mark.parametrize("arm", ["h-tree", "h-overlap", "h-local"])
def test_fresh_thermal_matched_initialization_and_native_shadow_step(arm):
    from channelthermal.data.collation import ChannelThermalBatchCollator
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.model import ChannelThermalHONFModel
    from channelthermal.plugin import create_plugin
    from channelthermal.training.campaign import copy_matched_physical_initial_state
    from channelthermal.training.campaign_work import CampaignForwardWork
    from channelthermal.training.epoch import make_model_inputs, run_epoch
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
    with CampaignForwardWork(model.core) as measured_work:
        output = hard_value_soft_hypergraph_forward(model, **inputs)
    work_snapshot = copy.deepcopy(measured_work.records)
    assert set(work_snapshot) == {"hard", "soft"}
    for mode in work_snapshot.values():
        assert set(mode) == {"P0", "P1", "P2"}
        assert all(phase["prepare_calls"] == 1 and phase["read_calls"] > 0 for phase in mode.values())
        assert all(phase["ledgers"]["hypergraph_MM_executed_rows"] > 0 for phase in mode.values())
    loss = (output["pred_field"] - batch["field_targets"]).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert measured_work.records == work_snapshot  # Forward scope excludes backward recomputation.
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
    if arm == "h-tree":
        model.set_training_progress(epoch=26, total_epochs=5000)
        native_loader = DataLoader(dataset, batch_size=1, collate_fn=ChannelThermalBatchCollator())
        metrics = run_epoch(model, native_loader, device, cfg["loss"], optimizer=optimizer, scaler=None,
                            amp=False, max_batches=1, local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
                            effective_internal_temperature_weight=1., effective_interface_weight=.2,
                            predicted_consistency_weight=.0005, gradient_clip_norm=1.,
                            campaign_config=cfg["training"]["campaign"], absolute_epoch=26,
                            forward_function=lambda **values: hard_value_soft_hypergraph_forward(model, **values))
        state = model.campaign_training_state
        assert state["calibration_policy_version"] == 2
        assert len(state["structural_calibration_samples"]) == 1
        assert not state["structural_calibration_complete"]
        assert metrics["campaign_structural_weight"] == 0.
        assert state["structural_calibration_samples"][0]["task_organizer_gradient_norm"] > 0
        print({"native_structural_calibration_sample": state["structural_calibration_samples"]})
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
        assert set(measured["response_forward_work"]) == {"hard", "soft"}
        assert all(phase["ledgers"]["hypergraph_QE_executed_rows"] > 0
                   for mode in measured["response_forward_work"].values() for phase in mode.values())
        (value_loss + auxiliary).backward()
        optimizer.step()
        print({"response_native_integration": measured})


@pytest.mark.skipif(not os.environ.get("HONF_ALIGNMENT_THERMAL_CHECKPOINT"), reason="Needs native local Thermal resources")
def test_native_mixed_module_batch_exact_denominator_loss_and_physical_gradient_parity():
    from channelthermal.data.collation import ChannelThermalBatchCollator, ModuleCountBucketBatchSampler
    from channelthermal.data.datasets import GlobalChannelThermalDataset
    from channelthermal.evaluation.loading import load_model
    from channelthermal.training.campaign import CampaignMicrobatchLoader
    from channelthermal.training.epoch import assemble_channelthermal_loss_terms, make_model_inputs

    device = torch.device("cpu")
    model, checkpoint = load_model(Path(os.environ["HONF_ALIGNMENT_THERMAL_CHECKPOINT"]), device)
    config = checkpoint["train_config"]
    dataset = GlobalChannelThermalDataset(config["dataset"]["packed_h5_path"], split="train", points_per_case=17,
        normalize_inputs=True, normalize_targets=True, random_point_sampling=True, seed=0, include_grid=False)
    dataset.set_epoch(26)
    sampler = ModuleCountBucketBatchSampler(dataset.selected_module_counts, batch_size=48, seed=0)
    sampler.set_epoch(26)
    native = next(iter(DataLoader(dataset, batch_sampler=sampler, collate_fn=ChannelThermalBatchCollator())))
    assert native["field_targets"].shape[:2] == (48, 17)
    assert len(set(native["module_count"].tolist())) > 1
    model.train()
    model.set_training_progress(epoch=100, total_epochs=5000)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def physical_loss(batch, normalization=None):
        output = model(**make_model_inputs(batch, local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
                                          return_predicted_port_outputs=True, return_port_global_consistency=True))
        return assemble_channelthermal_loss_terms(output, batch, model, config["loss"],
            local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
            effective_internal_temperature_weight=1., effective_interface_weight=.2,
            predicted_consistency_weight=.05, native_normalization_weights=normalization)["loss_physical"]

    def gradient(loss):
        values = torch.autograd.grad(loss, parameters, allow_unused=True)
        return torch.cat([torch.zeros(parameter.numel()) if value is None else value.reshape(-1)
                          for parameter, value in zip(parameters, values)])

    full_loss = physical_loss(native)
    reference_loss = float(full_loss.detach())
    reference_gradient = gradient(full_loss)

    class OneNativeBatch:
        dataset = None
        def __len__(self):
            return 1
        def __iter__(self):
            yield copy.deepcopy(native)

    seen, steps = [], 0
    accumulated_loss = 0.
    accumulated_gradient = torch.zeros_like(reference_gradient)
    for batch in CampaignMicrobatchLoader(OneNativeBatch(), 8, native_loss_denominators=True):
        normalization = batch.pop("_native_loss_normalization_weights")
        weight = batch.pop("_accumulation_weight")
        steps += int(batch.pop("_optimizer_boundary"))
        batch.pop("_optimizer_start")
        batch.pop("_auxiliary_due")
        seen.extend(batch["case_id"])
        value = physical_loss(batch, normalization) * weight
        accumulated_loss += float(value.detach())
        accumulated_gradient += gradient(value)
    assert seen == native["case_id"] and steps == 1
    assert accumulated_loss == pytest.approx(reference_loss, rel=2e-5, abs=2e-6)
    relative_gradient_error = torch.linalg.vector_norm(accumulated_gradient-reference_gradient) / torch.linalg.vector_norm(reference_gradient)
    assert float(relative_gradient_error) < 2e-5
    print({"native_M": native["module_count"].tolist(), "Q": 17, "full_loss": reference_loss,
           "exact_accumulated_loss": accumulated_loss, "physical_gradient_relative_error": float(relative_gradient_error)})
