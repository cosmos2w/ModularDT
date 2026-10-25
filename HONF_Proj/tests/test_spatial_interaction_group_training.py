from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from channelthermal.training.spatial_interaction_group_objective import (
    conditional_utility_huber,
    merge_named_gradients,
)
from channelthermal.training.spatial_interaction_group_task import (
    ThermalSpatialInteractionGroupTask,
    _normalizer_lineage_is_valid,
)
from honf_runtime.contextual_frontier_training import (
    _engine_config,
    _interaction_group_task_factory,
    _run_identity,
    _validate_loaded_contextual_config,
    load_contextual_frontier_config,
)


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT / "configs" / "research"


@pytest.mark.parametrize(
    ("arm", "kind", "membership", "first_segment"),
    (
        ("F13-0", "dev", "fixed25_train150_dev22", 100),
        ("F13-1", "dev", "fixed25_train150_dev22", 100),
        ("F13-0", "fulltrain", "full_train600", 200),
        ("F13-1", "fulltrain", "full_train600", 200),
    ),
)
def test_versioned_configs_bind_matching_spatial_runner_contract(
    arm: str, kind: str, membership: str, first_segment: int,
) -> None:
    config, _ = load_contextual_frontier_config(
        CONFIG_DIR / f"thermal_spatial_groups_{arm.lower().replace('-', '_')}_{kind}_v1.json"
    )
    section = config["interaction_groups"]
    assert section["arm"] == arm
    assert section["core_schema"] == "spatial_interaction_group_fields_v1"
    assert section["model"]["schema"] == "spatial_interaction_group_fields_v1"
    assert section["model"]["architecture"] == arm
    assert config["training"]["membership"] == membership
    assert config["training"]["first_segment_epochs"] == first_segment
    assert config["training"]["effective_batch_size"] == 48
    assert config["training"]["microbatch_cases"] == 24
    assert config["training"]["primary_queries"] == 1024
    assert config["training"]["curve_every"] == 10
    assert config["training"]["progress_enabled"] is True
    assert config["initialization"]["normalization_mode"].startswith("inherit_")
    assert config["objective"]["paired_utility_every_primary_updates"] == 4
    assert config["objective"]["paired_utility_physical_reduction"] == "average_over_actual_macro_case_count"
    assert config["objective"]["paired_utility_huber_coefficient"] == 1.0
    assert config["objective"]["response_coefficient_source"] == "literal_initialization_checkpoint"
    assert config["objective"]["response_microbatch_every_primary_updates"] == 4
    assert config["objective"]["scale_exploration_fraction"] == 0.25
    assert config["objective"]["additional_work_pressure"] == 0.0
    assert config["storage"]["output_root"].startswith("/data2/")
    assert _engine_config(config).checkpoint_epochs == tuple(config["training"]["checkpoint_epochs"])
    identity = _run_identity(config, SimpleNamespace())
    assert identity["workflow"] == "spatial_interaction_groups_joint_training_v1"
    assert identity["recipe"]["native_weights"] == config["objective"]["native_weights"]
    assert _interaction_group_task_factory(config).__name__ == "build_thermal_spatial_interaction_group_task"


def test_loaded_resume_config_rejects_changed_paired_reduction() -> None:
    config, _ = load_contextual_frontier_config(
        CONFIG_DIR / "thermal_spatial_groups_f13_0_dev_v1.json"
    )
    malformed = copy.deepcopy(config)
    malformed["objective"]["paired_utility_physical_reduction"] = "per_case_sum"
    with pytest.raises(ValueError, match="objective contract"):
        _validate_loaded_contextual_config(malformed)


def test_fulltrain_endpoint_proves_inherited_train150_normalizer_lineage() -> None:
    literal_parent = {
        "normalization_fit_scope": "selected TRAIN only",
    }
    development_endpoint = {
        "normalization_fit_scope": (
            "inherited from the literal epoch100 initialization checkpoint; no refit"
        ),
        "initialization": {
            "source_kind": "balanced_singleton_recovery_epoch100",
            "normalization_mode": "inherit_balanced_singleton_train150_stats",
            "normalization_scope": "selected TRAIN only",
        },
    }

    assert _normalizer_lineage_is_valid("balanced_singleton_recovery_epoch100", literal_parent)
    assert _normalizer_lineage_is_valid("same_model_development_epoch100", development_endpoint)
    development_endpoint["initialization"]["normalization_scope"] = "all source TRAIN"
    assert not _normalizer_lineage_is_valid("same_model_development_epoch100", development_endpoint)


class _UtilityCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.utility_head = torch.nn.Linear(3, 1)
        self.encoder_weight = torch.nn.Parameter(torch.ones(()))

    def predict_utility(self, features: torch.Tensor) -> torch.Tensor:
        return self.utility_head(features).squeeze(-1)


def test_conditional_huber_detaches_input_and_label_and_returns_head_only_gradients() -> None:
    core = _UtilityCore()
    features = torch.tensor([0.3, -0.8, 1.4], requires_grad=True)
    label = torch.tensor(0.7, requires_grad=True)

    prediction, loss, gradients = conditional_utility_huber(core, features, label)

    assert prediction.ndim == 0 and torch.isfinite(prediction)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {id(parameter) for parameter, _ in gradients} == {
        id(parameter) for parameter in core.utility_head.parameters()
    }
    assert all(torch.isfinite(gradient).all() for _, gradient in gradients)
    assert features.grad is None
    assert label.grad is None
    assert core.encoder_weight.grad is None

    named = {f"core.utility_head.{name}": parameter for name, parameter in core.utility_head.named_parameters()}
    merge_named_gradients(named, {
        f"core.utility_head.{name}": gradient
        for name, parameter in core.utility_head.named_parameters()
        for target, gradient in gradients
        if parameter is target
    })
    assert all(parameter.grad is not None for parameter in core.utility_head.parameters())


def test_paired_forced_frontiers_alternate_child_and_grandchild_descendants() -> None:
    parent0, child0, depth0 = ThermalSpatialInteractionGroupTask._pair_forced_frontiers(None, 17, 0)
    parent1, child1, depth1 = ThermalSpatialInteractionGroupTask._pair_forced_frontiers(None, 17, 1)
    assert parent0 == parent1 == {0: (17, "parent")}
    assert child0 == {0: (17, "children")}
    assert child1 == {0: (17, "grandchildren")}
    assert (depth0, depth1) == (1, 2)


def test_paired_probe_cycles_node_bands_across_module_count_strata() -> None:
    task = object.__new__(ThermalSpatialInteractionGroupTask)
    task._utility_stratum_counts = (3, 5, 7, 10)
    metadata = (
        {"node_id": 10, "depth": 1, "size": 128},
        {"node_id": 20, "depth": 3, "size": 32},
        {"node_id": 30, "depth": 6, "size": 8},
    )
    rows = [task._utility_node(metadata, index)[1] for index in range(12)]
    assert [row["candidate_band"] for row in rows] == (
        [[0, 2]] * 4 + [[1, 1]] * 4 + [[2, 0]] * 4
    )


def test_primary_frontier_exposure_ledger_separates_modes_k_and_depth() -> None:
    task = object.__new__(ThermalSpatialInteractionGroupTask)
    task.state = {"frontier_exposure_by_epoch": {}}
    task.model = SimpleNamespace(
        core=SimpleNamespace(organization_update=torch.tensor(2, dtype=torch.long))
    )

    def group(mode: str, *, nodes: tuple[int, ...], depths: tuple[int, ...], exploration_depth=None):
        selected = SimpleNamespace(
            selection_mode=mode,
            exploration_depth=exploration_depth,
            K=len(nodes),
            nodes=nodes,
            depths=depths,
        )
        return SimpleNamespace(frontier=selected)

    prepared = SimpleNamespace(organization=SimpleNamespace(layers=(
        (
            group("learned", nodes=(4, 7), depths=(0, 1, 1, 2, 2, 3, 3, 4)),
            group("scale_exploration", nodes=(0, 1, 2, 3), depths=(0, 1, 1, 2), exploration_depth=2),
        ),
    )))
    task._record_primary_frontier_exposure(prepared, epoch=1)

    epoch = task.state["frontier_exposure_by_epoch"]["1"]
    assert epoch["case_presentations"] == 2
    assert epoch["by_mode"]["learned"]["group_count_histogram"] == {"2": 1}
    assert epoch["by_mode"]["learned"]["group_depth_histogram"] == {"2": 1, "4": 1}
    assert epoch["by_mode"]["scale_exploration"]["exploration_depth_histogram"] == {"2": 1}
    assert epoch["by_mode"]["scale_exploration"]["group_count_histogram"] == {"4": 1}
    assert epoch["by_update"]["2"]["case_presentations"] == 2
