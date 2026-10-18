"""CPU torch exercises of the exact batched path used on the selected GPU."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hetmf.environment import generate_environment  # noqa: E402
from hetmf.gpu_physics import BatchSolveError, _heat_rhs, solve_gpu_batch  # noqa: E402
from hetmf.layouts import generate_layout  # noqa: E402
from hetmf.physics import heat_from_potential, prepare_scene, solve_cpu  # noqa: E402


def _scene(seed, count):
    return {
        "state_id": f"batch_{seed}_{count}",
        "environment": generate_environment(seed, "correlated_mosaic", contrast_s=8, rho=0.7),
        "contacts": generate_layout(seed + 29, count, "dispersed", "multilevel")["contacts"],
    }


def test_torch_face_energy_rhs_matches_independent_cpu_partition():
    scene = _scene(3, 3)
    prepared = prepare_scene(scene, 12)
    rng = np.random.default_rng(3)
    v = rng.normal(size=(12, 12))
    heat = heat_from_potential(prepared, v)
    result = _heat_rhs(
        torch.from_numpy(v[None]),
        torch.from_numpy(prepared["s"][None]),
        torch.from_numpy(prepared["face_s_x"][None]),
        torch.from_numpy(prepared["face_s_y"][None]),
        torch.from_numpy(prepared["contact_density"][None]),
        torch.from_numpy(prepared["u"][None]),
        prepared["area"],
    )
    np.testing.assert_allclose(result[0].numpy(), heat["rhs_t"], rtol=1e-14, atol=1e-14)


def test_batched_fp64_cpu_torch_matches_sparse_direct_and_balances():
    scenes = [_scene(5, 2), _scene(9, 4)]
    batched = solve_gpu_batch(scenes, 16, device="cpu", precision="fp64")
    assert len(batched) == len(scenes)
    for scene, output in zip(scenes, batched):
        reference = solve_cpu(scene, 16)
        np.testing.assert_allclose(output["v"], reference["v"], rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(output["theta"], reference["theta"], rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(output["terminal_currents"], reference["terminal_currents"],
                                   rtol=1e-7, atol=1e-8)
        assert output["diagnostics"]["electrical_solver"]["accepted"]
        assert output["diagnostics"]["thermal_solver"]["accepted"]
        assert output["diagnostics"]["batch_size"] == 2
        for name in ("terminal_current_balance", "electrical_energy_balance", "thermal_energy_balance"):
            assert abs(output["diagnostics"][name]["relative"]) < 1e-6


def test_max_iteration_failure_names_exact_state_without_returning_fields():
    scene = _scene(4, 2)
    with pytest.raises(BatchSolveError) as exc_info:
        solve_gpu_batch([scene], 12, device="cpu", max_iterations=0)
    error = exc_info.value
    assert error.stage == "electrical"
    assert error.failed_ids == [scene["state_id"]]
    assert error.reasons == ["max_iterations_or_true_residual"]
    assert error.details[0]["initial_iterations"] == 0
    assert not error.details[0]["accepted"]


def test_mixed_precision_batched_path_has_canonical_fp64_diagnostics():
    scene = _scene(11, 3)
    output = solve_gpu_batch([scene], 12, device="cpu", precision="mixed")[0]
    for stage in ("electrical_solver", "thermal_solver"):
        diagnostics = output["diagnostics"][stage]
        assert diagnostics["accepted"]
        assert diagnostics["precision"] == "mixed"
        assert diagnostics["true_residual_l2_integrated"] <= max(
            1e-10, 1e-6 * diagnostics["rhs_l2_integrated"])
