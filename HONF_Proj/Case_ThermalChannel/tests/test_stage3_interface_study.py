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
