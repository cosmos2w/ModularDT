"""Bounded paired-head contracts; small CPU/CUDA fixtures are not scientific runs."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_inverse_core.heat_generative_comparison import (
    PairedHeatHeads,
    PublicHeatTask,
    link_policy,
    observation_intervention,
    save_heat_draw,
    save_heat_payload,
    validate_review_stage,
)
from honf_inverse_core.models.centered_simplex_velocity import HeatSimplexCondition
from honf_inverse_core.models.frozen_packet_diffusion import PacketLinks, SpatialConditionalPacketDenoiser


def fixture():
    condition = HeatSimplexCondition(torch.zeros(1, 3, 1), torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool), torch.tensor([[[0., 0.], [1., 0.], [2., 0.]]]),
        torch.tensor([[[.5, 0., .2], [1.5, 0., .8]]]), torch.ones(1, 2, dtype=torch.bool),
        total_heat=torch.tensor([[3.]]))
    calls = []

    def provider(heat, public):
        calls.append(heat.clone())
        torch.testing.assert_close(heat.sum(1), public.total_heat)
        assert (heat >= 0).all()
        return PacketLinks(torch.tensor([[[0., 1., 0.], [1., 0., 0.], [0., 1., 0.]]]),
            torch.tensor([[[1., 0., 0.], [0., 1., 1.]]]), heat.expand(-1, -1, 2),
            torch.ones(1, 3, 2), torch.ones(1, 2, 3), torch.ones(1, 2, 2), torch.ones(1, 2, 2),
            torch.ones(1, 2, dtype=torch.bool), condition.module_features, condition.sensor_features[..., :2])

    def predictor(heat):
        return {"observed": heat[:, :2], "held": heat[:, 2:], "peaks": heat.squeeze(-1),
                "pressure": heat.sum(1)}

    task = PublicHeatTask("train_1", "train", condition, provider, predictor,
                         torch.tensor([[[.2], [.8]]]))
    torch.manual_seed(3)
    denoiser = SpatialConditionalPacketDenoiser(design_dim=1, module_dim=2, sensor_dim=3,
        embedding_dim=2, coordinate_dim=2, hidden_dim=8, layers=1)
    return task, torch.tensor([[[.2], [.8], [2.]]]), denoiser, calls


def test_paired_update_calls_one_provider_and_starts_equal():
    task, target, denoiser, calls = fixture()
    heads = PairedHeatHeads(denoiser, steps=2, seed=9)
    for name, weight in heads.graph.state_dict().items():
        torch.testing.assert_close(weight, heads.full.state_dict()[name])
    row = heads.step(task, target)
    assert len(calls) == 1 and row["update"] == 1
    assert not torch.equal(calls[0], target)
    assert row["graph_loss"] != row["full_loss"]
    assert all(parameter.grad is None for parameter in denoiser.parameters())


def test_restore_preserves_paired_noise_optimizer_and_next_update():
    task, target, denoiser, _ = fixture()
    first = PairedHeatHeads(denoiser, steps=2, seed=9)
    first.step(task, target)
    state = copy.deepcopy(first.checkpoint())
    restored = PairedHeatHeads(denoiser, steps=2, seed=300)
    restored.restore(state)
    expected, actual = first.step(task, target), restored.step(task, target)
    assert actual == expected
    for name, parameter in first.graph.state_dict().items():
        torch.testing.assert_close(parameter, restored.graph.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="schedule changed"):
        PairedHeatHeads(denoiser, steps=3).restore(state)


def assert_restored_state_equal(expected, actual):
    """Compare both head/optimizer values and devices after real serialization."""
    if isinstance(expected, torch.Tensor):
        assert actual.device == expected.device and actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_restored_state_equal(value, actual[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            assert_restored_state_equal(expected_item, actual_item)
    else:
        assert actual == expected


def assert_adamw_state_devices(heads, device):
    for name, model in heads.models.items():
        optimizer = heads.optimizers[name]
        assert optimizer.state  # Both optimizers have completed a real update.
        for parameter in model.parameters():
            assert parameter.device == device
        for parameter, state in optimizer.state.items():
            assert state["step"].device.type == "cpu"  # Default AdamW is not capturable/fused.
            for key in ("exp_avg", "exp_avg_sq"):
                assert state[key].device == parameter.device
                assert state[key].dtype == parameter.dtype


def fixture_on_device(device):
    from dataclasses import replace

    with torch.device(device):
        task, target, denoiser, calls = fixture()
    original_provider = task.provider

    def provider(heat, public):
        # Keep fixture-created provider tensors on the backend without moving
        # native AdamW's default noncapturable step counter off the CPU.
        with torch.device(device):
            return original_provider(heat, public)

    return replace(task, provider=provider), target, denoiser, calls


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable for the tiny serialized-resume fixture"))])
def test_serialized_paired_resume_preserves_cpu_tasks_and_both_head_updates(tmp_path, device):
    # Construction and provider-created tensors follow the same backend;
    # the independent native task-selection generator always remains on CPU.
    task, target, denoiser, _ = fixture_on_device(device)
    first = PairedHeatHeads(denoiser, steps=2, seed=9)
    task_stream = torch.Generator(device="cpu").manual_seed(10)
    torch.randint(600, (), generator=task_stream, device="cpu")
    first.step(task, target)
    path = tmp_path / "paired_resume.pt"
    save_heat_payload({"heads": first.checkpoint(), "task_stream": task_stream.get_state()}, path)

    saved = tool_module().load_paired_resume(path)
    assert saved["task_stream"].device.type == "cpu"
    assert saved["heads"]["generator_state"].device.type == "cpu"
    for model_state in saved["heads"]["models"].values():
        assert all(value.device.type == "cpu" for value in model_state.values())
    for optimizer_state in saved["heads"]["optimizers"].values():
        assert all(value.device.type == "cpu" for state in optimizer_state["state"].values()
                   for value in state.values() if isinstance(value, torch.Tensor))

    restored = PairedHeatHeads(denoiser, steps=2, seed=300)
    restored.restore(saved["heads"])
    restored_tasks = torch.Generator(device="cpu").manual_seed(301)
    restored_tasks.set_state(saved["task_stream"])
    assert_restored_state_equal(first.checkpoint(), restored.checkpoint())
    expected_index = torch.randint(600, (), generator=task_stream, device="cpu")
    actual_index = torch.randint(600, (), generator=restored_tasks, device="cpu")
    torch.testing.assert_close(actual_index, expected_index, atol=0, rtol=0)
    torch.testing.assert_close(restored_tasks.get_state(), task_stream.get_state(), atol=0, rtol=0)

    # Preview the next actual time/noise draws using independent generators
    # so the paired update stream itself is not advanced by the assertion.
    streams = []
    for heads in (first, restored):
        stream = torch.Generator(device=target.device)
        stream.set_state(heads.generator.get_state())
        streams.append((torch.rand((1,), generator=stream, device=target.device),
                        torch.randn(target.shape, generator=stream, device=target.device)))
    for expected, actual in zip(*streams, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected_row, actual_row = first.step(task, target), restored.step(task, target)
    assert actual_row == expected_row
    assert actual_row["time"] == streams[0][0].cpu().tolist()
    assert first.update == restored.update == 2
    assert_restored_state_equal(first.checkpoint(), restored.checkpoint())
    for heads in (first, restored):
        assert_adamw_state_devices(heads, target.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable for the mapped RNG fixture")
def test_restore_accepts_cuda_mapped_rng_without_remapping_cpu_model_optimizer_payload(tmp_path):
    task, target, denoiser, _ = fixture_on_device("cuda")
    first = PairedHeatHeads(denoiser, steps=2, seed=9)
    first.step(task, target)
    path = tmp_path / "paired_resume.pt"
    save_heat_payload({"heads": first.checkpoint()}, path)
    saved = tool_module().load_paired_resume(path)
    mapped_rng = copy.deepcopy(saved["heads"])
    mapped_rng["generator_state"] = mapped_rng["generator_state"].to(target.device)
    assert mapped_rng["generator_state"].is_cuda
    assert all(value.device.type == "cpu" for state in mapped_rng["models"].values()
               for value in state.values())
    restored = PairedHeatHeads(denoiser, steps=2, seed=300)
    restored.restore(mapped_rng)
    assert_restored_state_equal(first.checkpoint(), restored.checkpoint())
    assert_adamw_state_devices(restored, target.device)


def test_public_observation_interventions_preserve_locations_and_clean_heat_absent():
    task, _, _, _ = fixture()
    removed = observation_intervention(task.condition, "removed")
    shuffled = observation_intervention(task.condition, "shuffled")
    torch.testing.assert_close(removed.sensor_features[..., :2], task.condition.sensor_features[..., :2])
    assert not removed.sensor_channel_valid[..., 2:].any()
    assert not removed.sensor_features[..., 2:].any()
    torch.testing.assert_close(shuffled.sensor_features[..., 2:], task.condition.sensor_features.flip(1)[..., 2:])
    assert not task.condition.known_state.any()


def test_uniform_public_total_policy_reuses_one_hidden_target_free_stream():
    task, target, _, calls = fixture()
    provider = link_policy(task, "uniform_public_total")
    links = provider(target, task.condition)
    provider(torch.zeros_like(target), task.condition)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], torch.ones_like(target))
    torch.testing.assert_close(links.module_embeddings, torch.ones(1, 3, 2))


def test_update_and_conditioner_check_reject_development_and_cap():
    task, target, denoiser, _ = fixture()
    heads = PairedHeatHeads(denoiser, steps=2)
    from dataclasses import replace
    with pytest.raises(ValueError, match="train-only"):
        heads.step(replace(task, partition="development"), target)
    with pytest.raises(ValueError, match="training record"):
        heads.training_conditioner_check(replace(task, partition="development"), target)
    heads.update = 1500
    with pytest.raises(ValueError, match="1500"):
        heads.step(task, target)


def test_training_only_conditioner_probe_has_actual_sensor_gradient():
    task, target, denoiser, calls = fixture()
    heads = PairedHeatHeads(denoiser, steps=2)
    check = heads.training_conditioner_check(task, target)
    assert len(calls) == 1
    assert check["partition"] == "train"
    assert check["observation_value_gradient_norm"] > 0
    assert check["same_weight_graph_full_velocity_rms"] > 0
    assert check["loss"]["graph"]["original"] != check["loss"]["graph"]["removed"]


def test_completed_draw_is_saved_with_raw_projected_sensor_and_actual_time_axes(tmp_path):
    task, target, denoiser, _ = fixture()
    heads = PairedHeatHeads(denoiser, steps=2)
    path = tmp_path / "draw.pt"
    saved = save_heat_draw(heads.graph, task, target[:, 2:], initial_noise=torch.tensor([[[5.], [-5.], [0.]]]), path=path)
    assert path.exists() and saved["raw_heat"].shape == (3, 1, 3, 1)
    assert saved["state_timesteps"] == (1., .5, 0.)
    assert saved["clean_estimate_timesteps"] == (1., .5)
    assert saved["projection_mask"].any()
    assert (saved["projected_heat"] >= 0).all()
    torch.testing.assert_close(saved["projected_heat"].sum(2), torch.full((3, 1, 1), 3.))
    assert saved["observed_predictions"].shape == (3, 1, 2, 1)
    assert saved["held_predictions"].shape == (3, 1, 1, 1)
    assert len(saved["sampling_elapsed_seconds"]) == 3
    assert len(saved["typed_access_trail"]) == 2
    assert set(saved["typed_access_trail"][0]) >= {"module_source", "sensor_source", "module_environment", "environment_module", "sensor_environment"}


def test_review_stops_require_explicit_stage_continuation():
    for start, stop in ((0, 200), (200, 750), (750, 1500)):
        validate_review_stage(start, stop)
    for start, stop in ((0, 750), (0, 1500), (200, 1500), (1500, 1500)):
        with pytest.raises(ValueError, match="Review stages"):
            validate_review_stage(start, stop)


def test_failed_draw_write_preserves_previous_final_and_partial_evidence(tmp_path, monkeypatch):
    final = tmp_path / "draw.pt"
    final.write_bytes(b"previous completed draw")
    def fail(_payload, path):
        Path(path).write_bytes(b"partial write")
        raise OSError("No space left on device")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="No space"):
        save_heat_payload({}, final)
    assert final.read_bytes() == b"previous completed draw"
    assert final.with_name("draw.pt.tmp").read_bytes() == b"partial write"


def tool_module():
    path = Path(__file__).resolve().parents[1] / "tools/thermal_campaign_heat_generative.py"
    spec = importlib.util.spec_from_file_location("heat_generative_tool_test", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_public_template_strips_solved_data_and_preserves_only_geometry():
    module = tool_module()
    structure = {name: np.ones(1) for name in ("module_present", "material_params", "re", "u_in", "domain_length_x", "domain_length_y")}
    structure.update(module_centers=np.zeros((1, 2)), heat_powers=np.asarray([999.]))
    template = module.public_template({"structure": structure, "interface_condition": np.ones((1, 2, 8)),
        "module_internal_query_points": np.zeros((2, 2)), "steady_field": np.asarray([123.])})
    assert "heat_powers" not in template["structure"]
    assert "steady_field" not in template
    assert template["interface_condition"].shape == (1, 2, 3)
    assert module.parse_args(["--checkpoint", "a", "--dataset", "b", "--output-dir", "c", "--qualified-organizer", "reviewed"]).stop_update == 200
    features = module.public_module_features({"structure": structure}, total=3.)
    structure["heat_powers"] = np.asarray([.01])
    np.testing.assert_array_equal(features, module.public_module_features({"structure": structure}, total=3.))
    with pytest.raises(ValueError, match="600 training"):
        module.TrainOnlyNormalization.fit(type("WrongPartition", (), {"__len__": lambda self: 600, "split": "test"})())


def test_conditioner_probe_skips_single_module_and_zero_public_total():
    module = tool_module()
    records = [{"case_id": "one", "structure": {"module_present": np.asarray([1., 0.]), "heat_powers": np.asarray([2., 0.])}},
        {"case_id": "zero", "structure": {"module_present": np.asarray([1., 1.]), "heat_powers": np.asarray([0., 0.])}},
        {"case_id": "eligible", "structure": {"module_present": np.asarray([1., 1.]), "heat_powers": np.asarray([.2, 1.8])}}]
    dataset = type("Dataset", (), {"split": "train", "__len__": lambda self: len(records), "__getitem__": lambda self, index: records[index]})()
    assert module.eligible_training_probe(dataset)["case_id"] == "eligible"


def test_inverse_normalization_explicit_training_membership_preserves_full_default(monkeypatch):
    module = tool_module()
    monkeypatch.setattr(module, "sensor_panel", lambda sample: (
        np.zeros((14, 2)), np.arange(14), (), np.arange(6), np.arange(6, 12)))
    structure = {name: np.ones(1) for name in ("module_present", "material_params", "re", "u_in", "domain_length_x", "domain_length_y")}
    structure.update(module_centers=np.zeros((1, 2)), heat_powers=np.asarray([2.]))
    records = [{"case_id": case_id, "structure": structure, "steady_field": np.full((14, 1), value)}
               for case_id, value in (("train_a", 2.), ("train_b", 4.))]

    class Dataset:
        split = "train"
        channel_order = ("temperature",)

        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            return records[index]

    dataset = Dataset()
    normalization = module.TrainOnlyNormalization.fit(dataset, expected_case_ids=("train_a", "train_b"))
    assert normalization.case_ids == ("train_a", "train_b")
    assert normalization.sensor_mean[-1] == 3.
    assert normalization.metadata()["records"] == 2
    for ids in (("train_a", "held"), ("train_a", "train_a"), ()):
        with pytest.raises(ValueError, match="training case IDs|training records"):
            module.TrainOnlyNormalization.fit(dataset, expected_case_ids=ids)
    with pytest.raises(ValueError, match="600 training"):
        module.TrainOnlyNormalization.fit(dataset)
    dataset.split = "test"
    with pytest.raises(ValueError, match="training records"):
        module.TrainOnlyNormalization.fit(dataset, expected_case_ids=("train_a", "train_b"))


def test_generative_datasets_use_checkpoint_subset_and_shared_saved_normalizer(monkeypatch):
    module = tool_module()
    import thermal_development
    from channelthermal.data import datasets

    manifest = {"manifest_sha256": "a" * 64,
        "partitions": {"train": {"case_ids": ["train_a", "train_b"]}, "test": {"case_ids": ["held"]}}}
    resolutions, calls = [], []

    def resolve(config, path):
        resolutions.append((config, path))
        return manifest

    monkeypatch.setattr(thermal_development, "resolve_evaluation_manifest", resolve)
    monkeypatch.setattr(datasets, "GlobalChannelThermalDataset", lambda path, **kwargs: calls.append(kwargs) or kwargs)
    checkpoint = {"train_config": {"dataset": {"development_manifest_sha256": "a" * 64}},
                  "global_normalization_stats": {"field_mean_by_channel": [0.]}}
    train, dev, actual = module.build_generative_datasets(checkpoint, "fixture.h5")
    assert actual is manifest
    assert resolutions == [(checkpoint["train_config"]["dataset"], "fixture.h5")]
    assert train["case_ids"] == ("train_a", "train_b")
    assert dev["case_ids"] == ("held",)
    assert train["normalizer"] is dev["normalizer"]
    assert all(not row["normalize_inputs"] and not row["normalize_targets"] for row in calls)
    with pytest.raises(ValueError, match="selected-training"):
        module.build_generative_datasets({"train_config": checkpoint["train_config"]}, "fixture.h5")
    monkeypatch.setattr(thermal_development, "resolve_evaluation_manifest", lambda *args: None)
    calls.clear()
    module.build_generative_datasets({}, "fixture.h5")
    assert all("case_ids" not in row for row in calls)


def test_generative_output_accepts_ignored_data_symlink_and_rejects_source_escape(tmp_path, monkeypatch):
    module = tool_module()
    project = tmp_path / "project"
    ignored = project / "diagnostics/generated"
    ignored.mkdir(parents=True)
    data = tmp_path / "campaign_data"
    data.mkdir()
    (ignored / "campaign").symlink_to(data, target_is_directory=True)
    source = project / "src"
    source.mkdir()
    (ignored / "source_escape").symlink_to(source, target_is_directory=True)
    monkeypatch.setattr(module, "PROJECT_ROOT", project)
    assert module.resolve_evidence_output(ignored / "campaign/inverse/review_0200") == data / "inverse/review_0200"
    assert module.resolve_evidence_output(ignored / "ordinary") == ignored / "ordinary"
    for rejected in (source / "output", ignored / "../../src/output", data / "unanchored"):
        with pytest.raises(ValueError, match="ignored local output"):
            module.resolve_evidence_output(rejected)
    with pytest.raises(ValueError, match="non-output project path"):
        module.resolve_evidence_output(ignored / "source_escape/output")
    assert not (data / "inverse").exists()


def test_paired_review_checks_forward_state_and_labels_recovered_draw_scope(tmp_path):
    module = tool_module()
    forward = torch.nn.Linear(2, 1).eval().requires_grad_(False)
    snapshot = module.snapshot_forward_state(forward)
    result = module.record_frozen_review(forward, snapshot, tmp_path / "review_0200",
        update=200, draw_scope={"new_completed_draws": 0, "reused_saved_draws": 108})
    assert result["passed"] and result["reused_saved_draws"] == 108
    assert "not retrospectively certified" in result["verification_scope"]
    assert (tmp_path / "review_0200/forward_freeze_verification.json").is_file()
    with torch.no_grad():
        forward.weight.add_(1.)
    with pytest.raises(RuntimeError, match="state tensor changed: weight"):
        module.record_frozen_review(forward, snapshot, tmp_path / "review_0750",
            update=750, draw_scope={"new_completed_draws": 108, "reused_saved_draws": 0})
    assert not (tmp_path / "review_0750").exists()


def test_public_observation_task_needs_no_clean_heat_or_held_targets(monkeypatch):
    module = tool_module()
    forward = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    sample = {"structure": {"module_centers": np.asarray([[1., 2.], [3., 4.]]),
        "module_present": np.ones(2), "material_params": np.asarray([.018, .01, .02, 1., 1., .45]),
        "re": np.asarray([50.]), "u_in": np.asarray([1.]),
        "domain_length_x": np.asarray([12.]), "domain_length_y": np.asarray([6.])},
        "interface_condition": np.zeros((2, 4, 3)), "module_internal_query_points": np.zeros((3, 2))}
    captured = []

    def native_predictor(_model, _checkpoint, public, coordinates, observed, held):
        captured.append(public)
        assert "heat_powers" not in public["structure"] and "steady_field" not in public
        np.testing.assert_array_equal(public["structure"]["material_params"], sample["structure"]["material_params"])
        return (lambda heat: {"observed": heat.sum().expand(len(observed)),
                              "held": heat.sum().expand(len(held))}), torch.ones(2, dtype=torch.bool)

    monkeypatch.setattr(module, "native_heat_predictor", native_predictor)
    normalization = module.TrainOnlyNormalization(np.zeros(13), np.ones(13), np.zeros(3), np.ones(3), ())
    arguments = {"case_id": "stored_pair/baseline", "total": 3.,
        "coordinates": np.arange(28, dtype=np.float32).reshape(14, 2),
        "observed": np.arange(0, 12, 2), "held": np.arange(1, 12, 2),
        "observed_values": np.arange(6, dtype=np.float32)}
    task = module.build_public_observation_task(forward, {}, sample, "final_review_exposed", normalization, **arguments)
    assert len(captured) == 1 and not task.condition.known_state.any()
    assert task.partition == "final_review_exposed"
    torch.testing.assert_close(task.observed_reference[0, :, 0], torch.arange(6, dtype=torch.float32))
    torch.testing.assert_close(task.condition.module_features[0, :, -1], torch.full((2,), 3.))
    heat = torch.tensor([[[.5], [2.5]]], requires_grad=True)
    prediction = task.predictor(heat)
    prediction["held"].sum().backward()
    assert heat.grad is not None and (heat.grad != 0).all()
    with pytest.raises(ValueError, match="distinct observed/held"):
        module.build_public_observation_task(forward, {}, sample, "final_review_exposed", normalization,
            **{**arguments, "held": arguments["observed"]})
    with pytest.raises(ValueError, match="finite and valid"):
        module.build_public_observation_task(forward, {}, sample, "final_review_exposed", normalization,
            **{**arguments, "observed_values": np.full(6, np.nan)})


def test_valid_alternative_draws_share_noise_and_recovery_checks_physical_targets(tmp_path, monkeypatch):
    module = tool_module()
    original, _target, denoiser, _calls = fixture()
    heads = PairedHeatHeads(denoiser, steps=2, seed=8)
    pair = []
    tasks = {}
    for label, observation in (("baseline", .2), ("heat_transfer_plus", .4)):
        case_id = "stored_family:"+label
        public = SimpleNamespace(case_id=case_id,
            metadata={"family_id": "stored_family", "variant": label, "previously_exposed": True},
            public_sample={"structure": {"module_centers": np.zeros((3, 2)), "module_present": np.ones(3),
                "material_params": np.ones(6), "re": np.asarray([50.]), "u_in": np.asarray([1.]),
                "domain_length_x": np.asarray([12.]), "domain_length_y": np.asarray([6.])},
                "interface_condition": np.zeros((3, 4, 3)), "module_internal_query_points": np.zeros((2, 2))},
            source_module_ids=("m0", "m1", "m2"), source_id_to_slot={"m0": 0, "m1": 1, "m2": 2},
            sensor_names=("obs0", "obs1", "held0"), sensor_query_ids=("q0", "q1", "q2"),
            sensor_coordinates=np.asarray([[0., 0.], [1., 0.], [2., 0.]]),
            observed_rows=np.asarray([0, 1]), held_rows=np.asarray([2]), temperature_unit="fixture_temperature")
        hidden = SimpleNamespace(held_temperatures=np.asarray([.6]), heat=np.asarray([.2, .8, 2.]))
        condition = copy.deepcopy(original.condition)
        condition.sensor_features[..., 2] = observation
        tasks[case_id] = PublicHeatTask(case_id, "final_review_previously_exposed", condition,
            original.provider, original.predictor, torch.full((1, 2, 1), observation))
        pair.append((public, hidden))
    monkeypatch.setattr(module, "build_atlas_observation_task", lambda _model, _checkpoint, public, _normalization: tasks[public.case_id])
    args = SimpleNamespace(seed=11, embedding_policy="candidate_projected", evaluate_only=False, steps=2)
    result = module.evaluate_alternative_draws(heads, None, {}, None, tmp_path, args, [tuple(pair)])
    assert result == {"new_completed_draws": 4, "reused_saved_draws": 0}
    from honf_runtime.compat import load_trusted_checkpoint
    saved = [load_trusted_checkpoint(path, map_location="cpu") for path in sorted(tmp_path.rglob("draw_*.pt"))]
    assert len(saved) == 4
    assert all(torch.equal(saved[0]["initial_noise"], row["initial_noise"]) for row in saved)
    np.testing.assert_allclose(sorted(float(row["observed_reference"][0, 0, 0]) for row in saved), [.2, .2, .4, .4])
    args.evaluate_only = True
    assert module.evaluate_alternative_draws(heads, None, {}, None, tmp_path, args, [tuple(pair)]) == {
        "new_completed_draws": 0, "reused_saved_draws": 4}
    pair[1][1].held_temperatures = np.asarray([.7])
    with pytest.raises(ValueError, match="reference changed physical inputs or supervision"):
        module.evaluate_alternative_draws(heads, None, {}, None, tmp_path, args, [tuple(pair)])


def test_reference_recovery_rejects_changed_public_context_and_hidden_heat(tmp_path):
    module = tool_module()
    path = tmp_path / "evaluation_reference.pt"
    payload = {"public_geometry_context": {"re": torch.tensor([50.]), "material": torch.ones(6)},
               "source_id_to_slot": {"m0": 0, "m1": 1}, "reference_heat": torch.tensor([1., 2.])}
    module.save_or_validate_reference(payload, path)
    original_bytes = path.read_bytes()
    module.save_or_validate_reference(copy.deepcopy(payload), path)
    assert path.read_bytes() == original_bytes
    for key in ("context", "heat", "mapping"):
        changed = copy.deepcopy(payload)
        if key == "context":
            changed["public_geometry_context"]["re"] += 1
        elif key == "heat":
            changed["reference_heat"] = torch.tensor([2., 1.])
        else:
            changed["source_id_to_slot"] = {"m0": 1, "m1": 0}
        with pytest.raises(ValueError, match="reference changed physical inputs or supervision"):
            module.save_or_validate_reference(changed, path)
        assert path.read_bytes() == original_bytes


def test_saved_draw_recovery_rejects_changed_public_budget(tmp_path):
    module = tool_module()
    task, _target, denoiser, _calls = fixture()
    heads = PairedHeatHeads(denoiser, steps=2, seed=8)
    held, noise = torch.tensor([[[.6]]]), torch.zeros_like(task.condition.known_state)
    path = tmp_path / "draw.pt"
    args = SimpleNamespace(evaluate_only=False, steps=2)
    assert module.save_or_reuse_draw(heads.graph, task, held, noise=noise, path=path, dense=False,
        observation="original", provider=task.provider, args=args)
    from honf_runtime.compat import load_trusted_checkpoint
    saved = load_trusted_checkpoint(path, map_location="cpu")
    saved["public_total_heat"] += 1
    module.save_heat_payload(saved, path)
    args.evaluate_only = True
    with pytest.raises(ValueError, match="identity/integrity mismatch"):
        module.save_or_reuse_draw(heads.graph, task, held, noise=noise, path=path, dense=False,
            observation="original", provider=task.provider, args=args)


def test_recovery_does_not_recreate_missing_reference_beside_old_draws(tmp_path):
    module = tool_module()
    (tmp_path / "draw_00_graph.pt").write_bytes(b"previous draw")
    reference_path = tmp_path / "evaluation_reference.pt"
    with pytest.raises(ValueError, match="lack their original evaluation reference"):
        module.save_or_validate_reference({"public_total": torch.tensor([[3.]])}, reference_path)
    assert not reference_path.exists()
