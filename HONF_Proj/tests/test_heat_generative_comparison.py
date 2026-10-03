"""Bounded paired-head contracts; small CPU fixtures are not scientific runs."""

import copy
import importlib.util
from pathlib import Path

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
