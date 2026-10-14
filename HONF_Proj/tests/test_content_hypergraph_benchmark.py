from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))

from content_hypergraph_benchmark import (
    ARCHITECTURES,
    Arm,
    _model_state_cpu,
    _provider_identity_without_arm,
    _request_to_device,
    _state_sha256,
    _stress_core_probe,
    _stress_request,
    comparison_recipes,
    expected_optimizer_updates,
    initialization_receipt,
    is_oom_error,
    load_arms,
    oom_receipt,
    packing_independent_query_sha256,
    parse_int_list,
    validate_microbatch_packings,
)


class _ToyArm(nn.Module):
    def __init__(self, private_name: str, private_value: float) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 2, bias=False)
        self.register_parameter(private_name, nn.Parameter(torch.tensor([private_value], dtype=torch.float32)))


def _arm(architecture: str, model: nn.Module) -> Arm:
    return Arm(
        architecture=architecture,
        model=model,
        provider=None,
        recipe={"architecture": architecture},
        source="fresh_recipe",
        checkpoint=None,
        checkpoint_sha256=None,
        initial_model_state=_model_state_cpu(model),
    )


def test_optimizer_budget_counts_both_arms_and_full_epochs() -> None:
    assert expected_optimizer_updates(150, 48, epochs=5, packings=1) == 40
    assert expected_optimizer_updates(72, 24, epochs=5, packings=3) == 90
    assert set(ARCHITECTURES) == {"A-D", "A-H6"}


def test_state_digest_binds_scalar_routing_parameters_and_shape() -> None:
    state = {"live_temperature": torch.tensor(0.5), "enabled": torch.tensor(True)}
    original = _state_sha256(state)
    assert _state_sha256({name: value.clone() for name, value in state.items()}) == original
    assert _state_sha256({**state, "live_temperature": torch.tensor(0.6)}) != original
    assert _state_sha256({**state, "live_temperature": torch.tensor([0.5])}) != original


def test_complete_request_transfers_nested_prescribed_tensors_and_keeps_native_metadata() -> None:
    @dataclass(frozen=True)
    class NativeInputs:
        structure: dict
        case_ids: tuple[str, ...]

    native_ids = np.array([7, 11], dtype=np.int64)
    request = SimpleNamespace(
        scene_inputs=NativeInputs({"coordinates": torch.ones(1, 2, 2), "ids": native_ids}, ("0006",)),
        receivers={"heat": torch.ones(1, 2), "points": [torch.zeros(1, 3, 2)]},
        case_keys=("0006",),
    )
    moved = _request_to_device(request, "meta")
    assert isinstance(moved.scene_inputs, NativeInputs)
    assert moved.scene_inputs.structure["coordinates"].device.type == "meta"
    assert moved.receivers["heat"].device.type == "meta"
    assert moved.receivers["points"][0].device.type == "meta"
    assert moved.scene_inputs.structure["ids"] is native_ids
    assert moved.scene_inputs.case_ids == moved.case_keys == ("0006",)
    assert request.receivers["heat"].device.type == "cpu"


def test_reader128_cost_comparison_is_explicit_and_preserves_other_recipe_guards():
    ad = {"schema": "content_hypergraph_recipe_v1", "dataset": "thermal",
          "protocol": "development", "architecture": "A-D",
          "model": {"local_hidden_dim": 64, "thermal_response_rank": 4}, "seed": 0}
    h6 = {**ad, "architecture": "A-H6", "model": {**ad["model"], "local_hidden_dim": 128}}
    with pytest.raises(ValueError, match="recipes differ"):
        comparison_recipes(ad, h6)
    normalized = comparison_recipes(ad, h6, thermal_reader128_revision=True)
    assert normalized[0] == normalized[1] and h6["model"]["local_hidden_dim"] == 128
    for modified in ({**h6, "seed": 1},
                     {**h6, "model": {**h6["model"], "thermal_response_rank": 8}}):
        with pytest.raises(ValueError, match="recipes differ"):
            comparison_recipes(ad, modified, thermal_reader128_revision=True)
    with pytest.raises(ValueError, match="require Thermal v1 development"):
        comparison_recipes({**ad, "dataset": "wind"}, {**h6, "dataset": "wind"},
                           thermal_reader128_revision=True)


def _round2_recipe(name: str) -> dict:
    path = PROJECT / "src" / "config_core" / "forward" / "content_hypergraph" / name
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    ("dataset", "direct_name", "organized_name"),
    [
        ("thermal", "thermal_a2-d_development.json", "thermal_a2-h6_r1_centroids_development.json"),
        ("wind", "wind_a2-d_development.json", "wind_a2-h6_r1_centroids_development.json"),
    ],
)
def test_typed_centroid_cost_comparison_accepts_retained_r1_pair_for_both_datasets(
    dataset: str, direct_name: str, organized_name: str,
) -> None:
    direct, organized = _round2_recipe(direct_name), _round2_recipe(organized_name)
    original_organized = json.loads(json.dumps(organized))

    normalized_direct, normalized_organized = comparison_recipes(
        direct, organized, typed_centroid_revision=True,
    )

    assert direct["dataset"] == organized["dataset"] == dataset
    assert normalized_direct == normalized_organized
    assert organized == original_organized
    assert "query_edge_geometry_mode" not in normalized_organized["model"]


@pytest.mark.parametrize("dataset", ["thermal", "wind"])
def test_typed_centroid_comparison_rejects_non_single_field_or_non_development_changes(dataset: str) -> None:
    if dataset == "thermal":
        direct = _round2_recipe("thermal_a2-d_development.json")
        organized = _round2_recipe("thermal_a2-h6_r1_centroids_development.json")
    else:
        direct = _round2_recipe("wind_a2-d_development.json")
        organized = _round2_recipe("wind_a2-h6_r1_centroids_development.json")

    bad_dimension = json.loads(json.dumps(organized))
    bad_dimension["model"]["context_dim"] = 64
    with pytest.raises(ValueError):
        comparison_recipes(direct, bad_dimension, typed_centroid_revision=True)

    bad_flag = json.loads(json.dumps(organized))
    bad_flag["model"]["unsupported_reader_flag"] = True
    with pytest.raises(ValueError):
        comparison_recipes(direct, bad_flag, typed_centroid_revision=True)

    bad_direct_mode = json.loads(json.dumps(direct))
    bad_direct_mode["model"]["query_edge_geometry_mode"] = "typed_w_centroids_v1"
    with pytest.raises(ValueError):
        comparison_recipes(bad_direct_mode, organized, typed_centroid_revision=True)

    bad_loss = json.loads(json.dumps(organized))
    bad_loss["native_loss_override"] = {"field": 0.5}
    with pytest.raises(ValueError, match="losses or objectives"):
        comparison_recipes(direct, bad_loss, typed_centroid_revision=True)

    bad_protocol = json.loads(json.dumps(organized))
    bad_protocol["protocol"] = "full_train1000"
    with pytest.raises(ValueError, match="development"):
        comparison_recipes(direct, bad_protocol, typed_centroid_revision=True)


def test_typed_centroid_revision_is_mutually_exclusive_with_v1_reader_capacity_revision() -> None:
    direct = _round2_recipe("thermal_a2-d_development.json")
    organized = _round2_recipe("thermal_a2-h6_r1_centroids_development.json")
    with pytest.raises(ValueError, match="mutually exclusive"):
        comparison_recipes(
            direct, organized, thermal_reader128_revision=True, typed_centroid_revision=True,
        )


@pytest.mark.parametrize(
    ("dataset", "direct_name", "organized_name"),
    [
        ("thermal", "thermal_a2-d_development.json", "thermal_a2-h6_r1_centroids_development.json"),
        ("wind", "wind_a2-d_development.json", "wind_a2-h6_r1_centroids_development.json"),
    ],
)
def test_typed_centroid_load_arms_accepts_only_retained_pair_for_each_dataset(
    tmp_path: Path, monkeypatch, dataset: str, direct_name: str, organized_name: str,
) -> None:
    import content_hypergraph_benchmark as benchmark

    direct_recipe, organized_recipe = _round2_recipe(direct_name), _round2_recipe(organized_name)
    direct_path, organized_path = tmp_path / "direct.pt", tmp_path / "organized.pt"
    direct_path.write_bytes(b"fixture direct checkpoint")
    organized_path.write_bytes(b"fixture organized checkpoint")
    by_path = {direct_path.resolve(): ("A2-D", direct_recipe), organized_path.resolve(): ("A2-H6", organized_recipe)}

    class Provider:
        def __init__(self, architecture: str, recipe: dict) -> None:
            self.architecture = architecture
            self.recipe = recipe

        def identity_payload(self) -> dict:
            return {
                "architecture": self.architecture,
                "model_config": {"architecture": self.architecture},
                "recipe": self.recipe,
                "adapter_state": {"core": {"architecture": self.architecture}, "fixed_data": "same"},
            }

    def fake_loader(path: Path, device: str):
        del device
        architecture, recipe = by_path[path.resolve()]
        payload = {"arm": architecture, "experiment_identity": {"recipe": recipe}}
        return nn.Linear(2, 2), Provider(architecture, recipe), payload

    monkeypatch.setattr(benchmark, "load_evaluation_components", fake_loader)
    arms, _, _ = load_arms(
        dataset=dataset,
        device="cpu",
        recipe_path=None,
        checkpoint_ad=direct_path,
        checkpoint_ah6=organized_path,
        typed_centroid_revision=True,
    )

    assert set(arms) == {"A2-D", "A2-H6"}
    assert all(arm.source == "retained_checkpoint" for arm in arms.values())


@pytest.mark.parametrize("mode", ["fresh_recipe", "training_packings", "reader128"])
def test_typed_centroid_cli_rejects_nonretained_or_mixed_revision_before_output(
    tmp_path: Path, mode: str,
) -> None:
    import content_hypergraph_benchmark as benchmark

    output = tmp_path / mode
    args = ["--dataset", "thermal", "--output", str(output), "--typed-centroid-revision"]
    if mode == "fresh_recipe":
        args.extend(["--recipe", str(tmp_path / "unused.json")])
    else:
        args.extend(["--checkpoint-ad", str(tmp_path / "direct.pt"),
                     "--checkpoint-ah6", str(tmp_path / "organized.pt")])
    if mode == "training_packings":
        args.extend(["--training-packings", "8", "--optimizer-update-budget", "1"])
    elif mode == "reader128":
        args.append("--thermal-reader128-revision")

    with pytest.raises(SystemExit):
        benchmark.main(args)
    assert not output.exists()


def test_microbatch_packings_are_unique_and_bounded_by_effective_batch() -> None:
    assert parse_int_list("8,16,48", name="packings") == (8, 16, 48)
    assert validate_microbatch_packings((8, 16, 48), 48) == (8, 16, 48)
    with pytest.raises(ValueError, match="At most three"):
        validate_microbatch_packings((2, 4, 8, 16), 48)
    with pytest.raises(ValueError, match="unique values"):
        validate_microbatch_packings((8, 8), 48)
    with pytest.raises(ValueError, match="unique values"):
        validate_microbatch_packings((49,), 48)
    with pytest.raises(argparse.ArgumentTypeError, match="unique integers"):
        parse_int_list("8,8", name="packings")


def test_oom_receipt_keeps_fixed_work_point_and_has_no_retry() -> None:
    error = RuntimeError("CUDA out of memory while allocating tensor")
    receipt = oom_receipt("A-H6", 8192, error)
    assert is_oom_error(error)
    assert receipt["status"] == "oom"
    assert receipt["batch_size"] == 8
    assert receipt["requested_q_per_scene"] == 8192
    assert receipt["optimizer_updates"] == 0
    assert receipt["retry_with_smaller_batch"] is False
    assert not is_oom_error(RuntimeError("unrelated failure"))


def test_fresh_pair_binds_common_named_initialization_and_distinct_private_states() -> None:
    torch.manual_seed(3)
    ad_model = _ToyArm(private_name="direct_reader", private_value=1.0)
    torch.manual_seed(3)
    h6_model = _ToyArm(private_name="edge_reader", private_value=2.0)
    arms = {"A-D": _arm("A-D", ad_model), "A-H6": _arm("A-H6", h6_model)}

    receipt = initialization_receipt(arms, require_shared_match=True)

    assert receipt["shared_parameters_match"] is True
    assert receipt["common_named_parameter_count"] == 1
    assert receipt["common_named_parameter_sha256"]
    assert receipt["arm_state_sha256"]["A-D"] != receipt["arm_state_sha256"]["A-H6"]


def test_fresh_pair_rejects_different_common_initialization() -> None:
    torch.manual_seed(3)
    ad_model = _ToyArm(private_name="direct_reader", private_value=1.0)
    torch.manual_seed(4)
    h6_model = _ToyArm(private_name="edge_reader", private_value=2.0)
    arms = {"A-D": _arm("A-D", ad_model), "A-H6": _arm("A-H6", h6_model)}

    with pytest.raises(ValueError, match="shared initialization differs"):
        initialization_receipt(arms, require_shared_match=True)


def test_checkpoint_pair_identity_normalizer_removes_only_arm_specific_configuration() -> None:
    identity = {
        "architecture": "A-D",
        "model_config": {"architecture": "A-D"},
        "recipe": {"architecture": "A-D", "dataset": "wind"},
        "dataset": "thermal",
        "normalizer": {"field_std": [1.0, 2.0]},
        "adapter_state": {
            "core": {"architecture": "A-D", "width": 128},
            "training_data": {"manifest": "fixed25_v1"},
        },
    }
    normalized = _provider_identity_without_arm(identity)
    assert normalized == {
        "dataset": "thermal",
        "recipe": {"dataset": "wind"},
        "normalizer": {"field_std": [1.0, 2.0]},
        "adapter_state": {"training_data": {"manifest": "fixed25_v1"}},
    }
    assert identity["adapter_state"]["core"]["architecture"] == "A-D"


def test_wind_query_stream_digest_is_independent_of_microbatch_boundaries() -> None:
    coordinates = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    features = np.arange(4 * 3 * 7, dtype=np.float32).reshape(4, 3, 7)
    flat_indices = np.arange(12, dtype=np.int64).reshape(4, 3)
    role_ids = np.tile(np.array([0, 1, 2], dtype=np.int8), (4, 1))
    local_rows = np.array([0, 1, 1, 2, 3], dtype=np.int64)
    local_queries = np.array([0, 1, 2, 1, 2], dtype=np.int64)
    local_modules = np.array([2, 0, 1, 1, 0], dtype=np.int64)

    def batch(start: int, stop: int) -> SimpleNamespace:
        selected = (local_rows >= start) & (local_rows < stop)
        receivers = SimpleNamespace(
            coordinates_D=coordinates[start:stop],
            features=features[start:stop],
            flat_indices=flat_indices[start:stop],
            role_ids=role_ids[start:stop],
            local_batch_indices=local_rows[selected] - start,
            local_query_indices=local_queries[selected],
            local_module_indices=local_modules[selected],
        )
        return SimpleNamespace(receivers=receivers, case_keys=tuple(range(start, stop)))

    packed = [batch(0, 4)]
    split = [batch(0, 1), batch(1, 3), batch(3, 4)]
    assert packing_independent_query_sha256(packed, "wind") == packing_independent_query_sha256(split, "wind")


def test_thermal_stress_converts_native_scene_and_preserves_surface_support():
    from channelthermal.content_hypergraph import ThermalContentHypergraphAdapter
    from channelthermal.training.content_hypergraph_task import (
        ThermalContentReceivers,
        ThermalContentSceneInputs,
    )

    model = ThermalContentHypergraphAdapter({
        "field_mean_by_channel": [0.0] * 5, "field_std_by_channel": [1.0] * 5,
        "interface_targets_mean": [0.0, 0.0], "interface_targets_std": [1.0, 1.0],
        "internal_temperature_mean": [0.0], "internal_temperature_std": [1.0],
    }, nx=8, ny=6, core_config={
        "architecture": "A-H6", "context_dim": 16, "num_heads": 4,
        "context_depth": 1, "num_hyperedges": 6, "thermal_response_rank": 4,
        "local_hidden_dim": 8, "init_seed": 7,
    }).eval()
    structure = {
        "module_centers": torch.tensor([[[2.0, 2.0], [7.5, 4.0]]]),
        "module_present": torch.ones(1, 2),
        "module_source_ids": torch.tensor([[31, 47]]),
        "material_params": torch.tensor([[0.02, 0.01, 0.02, 1.0, 2.0, 0.22]]),
        "re": torch.tensor([[50.0]]), "u_in": torch.ones(1, 1),
        "domain_length_x": torch.tensor([[12.0]]), "domain_length_y": torch.tensor([[6.0]]),
    }
    fluid = torch.tensor([[[1.0, 1.0], [4.0, 3.0]]])
    material = torch.zeros(1, 2, 2, 2)
    native = SimpleNamespace(
        case_keys=("fixture",),
        scene_inputs=ThermalContentSceneInputs(structure, None, ("fixture",)),
        receivers=ThermalContentReceivers(fluid, material, torch.ones(1, 2), ntheta=64),
    )
    arm = Arm("A-H6", model, SimpleNamespace(make_scene=lambda inputs: inputs),
              {"dataset": "thermal"}, "test", None, None, _model_state_cpu(model))
    _, receivers = _stress_request(arm, native, batch_size=8, query_count=17)
    assert receivers.ntheta == 64
    assert receivers.fluid_xy.shape == (8, 17, 2)
    with torch.no_grad():
        result = _stress_core_probe(arm, native, query_count=19)
    assert result["batch_size"] == 8
    assert result["executed_receiver_rows"] == 8 * 19
    assert result["padded_edge_rows"] == 8 * 19 * 6
    assert result["native_grid_accuracy_claim"] is False
    model.zero_grad(set_to_none=True)
    backward_result = _stress_core_probe(arm, native, query_count=19, backward=True)
    assert backward_result["backward_completed"] and backward_result["finite_gradients"]
    assert backward_result["gradient_tensor_count"] > 0
