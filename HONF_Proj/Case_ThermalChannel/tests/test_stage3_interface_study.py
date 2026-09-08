"""Focused CPU tests for the Stage-3 interface-study orchestrator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "diagnostics" / "run_stage3_interface_study.py"
SPEC = importlib.util.spec_from_file_location("stage3_interface_study", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
stage3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage3
SPEC.loader.exec_module(stage3)


def _dummy_model() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            core_honf=SimpleNamespace(
                module_radius=0.1,
                domain_length_x=2.0,
                domain_length_y=1.0,
            )
        )
    )


def _sample() -> dict[str, object]:
    return {
        "case_id": "synthetic",
        "x_grid": np.asarray([[0.25, 1.0]], dtype=np.float32),
        "y_grid": np.asarray([[0.5, 0.5]], dtype=np.float32),
        "structure": {
            "module_centers": np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=np.float32),
            "module_present": np.asarray([1.0, 1.0], dtype=np.float32),
        },
    }


def test_checkpoint_labels_and_plan_shapes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    parsed = stage3.parse_checkpoint_specs([f"primary={checkpoint}"])
    assert parsed[0].label == "primary"
    assert parsed[0].path == checkpoint.resolve()
    assert stage3.parse_shape("8,192,8192") == (8, 192, 8192)
    assert stage3.DEFAULT_SCALING_SHAPES == (
        (8, 192, 8192),
        (32, 768, 8192),
        (32, 768, 65536),
        (32, 768, 262144),
        (128, 3072, 262144),
    )
    with pytest.raises(ValueError):
        stage3.parse_shape("8,192")


def test_duplicate_environment_preserves_token_values() -> None:
    environment = stage3.ChannelThermalEnvironment(
        env_coords=torch.arange(6, dtype=torch.float32).reshape(1, 3, 2),
        env_features=torch.arange(21, dtype=torch.float32).reshape(1, 3, 7),
    )
    duplicated = stage3.duplicate_environment_bundle(environment, factor=2)
    assert duplicated.env_coords.shape == (1, 6, 2)
    assert duplicated.env_features.shape == (1, 6, 7)
    assert torch.equal(duplicated.env_coords[:, :3], duplicated.env_coords[:, 3:])
    assert torch.equal(duplicated.env_features[:, :3], duplicated.env_features[:, 3:])
    with pytest.raises(ValueError):
        stage3.duplicate_environment_bundle(environment, factor=1)


def test_duplicated_environment_proxy_preserves_query_features() -> None:
    environment = stage3.ChannelThermalEnvironment(
        env_coords=torch.zeros(1, 3, 2),
        env_features=torch.zeros(1, 3, 7),
    )

    class Builder:
        def __call__(self, **kwargs):
            del kwargs
            return environment

        def query_features(self, value):
            return value + 1

    model = SimpleNamespace(environment_builder=Builder())
    original = model.environment_builder
    with stage3.duplicated_environment(model, factor=2):
        assert model.environment_builder().env_coords.shape == (1, 6, 2)
        assert model.environment_builder.query_features(2) == 3
    assert model.environment_builder is original


def test_interface_read_role_marker_is_scoped() -> None:
    from channelthermal import interface_field_coupling as coupling

    model = SimpleNamespace(core=SimpleNamespace())
    assert not hasattr(model.core, "_interface_read_role")
    with coupling._interface_read_role(model, "p1_refinement"):
        assert model.core._interface_read_role == "p1_refinement"
    assert not hasattr(model.core, "_interface_read_role")


def test_audit_read_summary_separates_supported_and_null_mass() -> None:
    summary = stage3._audit_read_summary(
        {
            "group_read_degree": torch.tensor([[2.0, 0.0]]),
            # B is the raw incidence factor; availability must use aB from
            # the explicitly exported effective geometric slots.
            "group_read_geometric_weight": torch.tensor([[[0.9, 0.9], [0.0, 0.0]]]),
            "group_read_effective_geometric_weight": torch.tensor(
                [[[0.2, 0.3], [0.0, 0.0]]]
            ),
            "group_read_weight_mass": torch.tensor([[0.4, 0.0]]),
            "group_read_conditional_mixture_norm": torch.tensor([[1.5, 0.0]]),
        }
    )
    assert summary["available"] is True
    assert summary["unsupported_receiver_fraction"] == 0.5
    assert summary["availability"]["all"]["mean"] == pytest.approx(0.25)
    assert summary["null_mass"]["supported"]["mean"] == pytest.approx(0.6)


def test_audit_read_summary_omits_availability_without_effective_geometry() -> None:
    summary = stage3._audit_read_summary(
        {
            "group_read_degree": torch.tensor([[1.0]]),
            # Raw B alone is not an availability measurement.
            "group_read_geometric_weight": torch.tensor([[[0.9]]]),
        }
    )
    assert "availability" not in summary


def test_audit_read_summary_excludes_padded_receivers_and_slots() -> None:
    summary = stage3._audit_read_summary(
        {
            "group_read_degree": torch.tensor([[2.0, 0.0]]),
            "group_read_group_index": torch.tensor([[[7, -1], [-1, -1]]]),
            "group_read_logit_mean": torch.tensor([[4.0, 100.0]]),
            "group_read_logit_std": torch.tensor([[0.5, 9.0]]),
            "group_read_logit": torch.tensor([[[10.0, 0.0], [0.0, 0.0]]]),
            "group_read_dot_product": torch.tensor([[[10.0, 0.0], [0.0, 0.0]]]),
            "main_context_norm": torch.tensor([[1.0, 99.0]]),
        },
        active_receiver_mask=torch.tensor([[True, False]]),
    )
    assert summary["receiver_count"] == 1
    assert summary["execution_receiver_count"] == 2
    assert summary["padded_receiver_count"] == 1
    assert summary["unsupported_receiver_fraction"] == 0.0
    assert summary["logits"]["all"]["mean"] == pytest.approx(4.0)
    assert summary["logit_spread"]["all"]["mean"] == pytest.approx(0.5)
    assert summary["main_context_norm"]["all"]["mean"] == pytest.approx(1.0)
    # The per-slot fallback is filtered by group index, so its reserved zero
    # slot does not dilute the observed logit.
    assert summary["dot_product_logits"]["all"]["mean"] == pytest.approx(10.0)


def test_phase_execution_and_incremental_comparison_are_explicit() -> None:
    outputs = {
        "interaction_aux": {
            "initial_port_group_read_degree": torch.ones(1, 2),
            "group_read_degree": torch.ones(1, 4),
        },
        "provisional_read_aux": {"group_read_degree": torch.ones(1, 2)},
    }
    assert stage3._phase_execution_summary(outputs) == {
        "P0_initial_port": True,
        "P1_refinement": True,
        "P2_field": True,
    }
    normal = {"pred_field": torch.tensor([10.0])}
    p0 = {"pred_field": torch.tensor([8.0])}
    p0_p1 = {"pred_field": torch.tensor([6.0])}
    relative_to_normal = stage3._summarize_outputs(normal, p0_p1)
    incremental = stage3._summarize_outputs(p0, p0_p1)
    assert relative_to_normal["pred_field"]["mean_abs"] == pytest.approx(4.0)
    assert incremental["pred_field"]["mean_abs"] == pytest.approx(2.0)
    assert incremental["pred_field"]["max_relative"] == pytest.approx(0.25)


def test_vector_response_norm_summary_respects_topology_masks() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    connected = torch.tensor([[True, False, True, False]])
    assert stage3._vector_response_norm_summary(values, connected) == {
        "count": 2,
        "mean": pytest.approx(2.0),
        "p95": pytest.approx(2.9),
        "max": pytest.approx(3.0),
    }
    assert stage3._vector_response_norm_summary(values, ~connected) == {
        "count": 2,
        "mean": pytest.approx(3.0),
        "p95": pytest.approx(3.9),
        "max": pytest.approx(4.0),
    }


def test_support_counts_separate_actual_degree_from_routing_slot_capacity() -> None:
    counts = stage3._support_counts(
        {"interaction_aux": {"support_centres": torch.zeros(3, 2)}},
        {
            "group_read_degree": torch.tensor([[2.0, 0.0, 3.0]]),
            "group_read_group_index": torch.full((1, 3, 16), -1, dtype=torch.long),
        },
    )
    assert counts["query_group_read_value_count"] == 5
    assert counts["query_group_read_degree_sum"] == 5
    assert counts["query_group_read_slot_capacity"] == 48
    assert counts["query_group_read_degree_mean"] == pytest.approx(5.0 / 3.0)
    assert counts["query_group_read_degree_max"] == pytest.approx(3.0)


def test_synthetic_scaling_aggregates_degree_and_preserves_chunk_configuration() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.core = SimpleNamespace(receiver_chunk_size=128)
            self.decode_chunks: list[tuple[int, int | None, bool]] = []

        def __call__(self, structure, query_xy, **kwargs):
            del structure, query_xy
            assert kwargs["return_routing_maps"] is False
            prepared = SimpleNamespace(
                prepared=SimpleNamespace(
                    backend_state=SimpleNamespace(
                        cache=SimpleNamespace(layout=SimpleNamespace(spatial_dimension=2))
                    )
                )
            )
            return {"prepared_state": prepared, "routing_aux": {}}

        def decode_prepared(self, prepared, query_xy, **kwargs):
            del prepared
            self.decode_chunks.append(
                (int(query_xy.shape[1]), kwargs["receiver_chunk_size"], kwargs["return_routing_maps"])
            )
            return {
                "pred_field": query_xy.new_zeros((query_xy.shape[0], query_xy.shape[1], 1)),
                "group_read_degree": query_xy.new_full((query_xy.shape[0], query_xy.shape[1]), 2.0),
            }

    model = FakeModel()
    query_xy = torch.zeros(1, 5, 2)
    prediction, _prepared, extras = stage3._prepared_synthetic_forward(
        model,
        {},
        query_xy,
        torch.device("cpu"),
        query_batch_size=2,
        receiver_chunk_size=2048,
        return_routing_maps=False,
    )
    assert prediction.shape == (1, 5, 1)
    assert extras["support"]["query_group_read_value_count"] == 10.0
    assert extras["support"]["query_group_read_degree_sum"] == 10.0
    assert extras["support"]["query_group_read_slot_capacity"] == 80.0
    assert model.core.receiver_chunk_size == 128
    assert model.decode_chunks == [(2, 2048, False), (2, 2048, False), (1, 2048, False)]


def test_geometry_perturbation_is_copied_and_valid() -> None:
    sample = _sample()
    model = _dummy_model()
    result = stage3.perturb_sample(sample, model, 0, (1.0, 0.0), 0.1)
    assert np.allclose(sample["structure"]["module_centers"], [[0.5, 0.5], [1.5, 0.5]])
    assert np.allclose(result["structure"]["module_centers"], [[0.6, 0.5], [1.5, 0.5]])
    assert stage3._family_module_indices(sample, "pair_separated") == (0, 1)
    assert stage3._family_module_indices(sample, "pair_crowded") == (0, 1)
    assert stage3._family_module_indices(sample, "triple") is None


@pytest.mark.parametrize(
    "task,arguments",
    [
        ("interventions", ["--sparse-checkpoint", "s=/tmp/s.pt"]),
        ("checkpoint_audit", ["--sparse-checkpoint", "a=/tmp/a.pt"] * 3),
        ("gradients", ["--sparse-checkpoint", "s=/tmp/s.pt"]),
        ("quadrature", ["--checkpoint", "a=/tmp/a.pt"] * 4),
        ("scaling", ["--checkpoint", "a=/tmp/a.pt"] * 3),
        ("requests", ["--primary-checkpoint", "p=/tmp/p.pt"]),
    ],
)
def test_parser_has_explicit_output_and_task_entrypoint(task: str, arguments: list[str]) -> None:
    parser = stage3.build_parser()
    parsed = parser.parse_args([task, *arguments, "--output", "/tmp/stage3.json"])
    assert parsed.task == task
    assert callable(parsed.handler)
    assert str(parsed.output) == "/tmp/stage3.json"


def test_scaling_parser_accepts_bounded_checkpoint_count_and_runtime_controls() -> None:
    parser = stage3.build_parser()
    parsed = parser.parse_args(
        [
            "scaling",
            "--checkpoint",
            "candidate=/tmp/candidate.pt",
            "--shape",
            "128,3072,262144",
            "--receiver-chunk-size",
            "2048",
            "--routing-mode",
            "summary",
            "--output",
            "/tmp/stage3.json",
        ]
    )
    assert parsed.receiver_chunk_size == 2048
    assert parsed.routing_mode == "summary"
    assert parsed.shape == ["128,3072,262144"]
