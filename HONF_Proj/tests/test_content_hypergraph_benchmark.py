from __future__ import annotations

import argparse
import sys
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
    _stress_core_probe,
    _stress_request,
    comparison_recipes,
    expected_optimizer_updates,
    initialization_receipt,
    is_oom_error,
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


def test_reader128_cost_comparison_is_explicit_and_preserves_other_recipe_guards():
    ad = {"dataset": "thermal", "protocol": "development", "architecture": "A-D",
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
    with pytest.raises(ValueError, match="require Thermal development"):
        comparison_recipes({**ad, "dataset": "wind"}, {**h6, "dataset": "wind"},
                           thermal_reader128_revision=True)


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
