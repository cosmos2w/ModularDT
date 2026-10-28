"""Compare split and uninterrupted A/L recovery with the formal schedule."""
import pytest
import torch

from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.unified_training import OptimizerGroupSpec, ScheduleSpec
from tests import test_r13_8_shared_engine_resume as baseline


def _constant_groups(self, model, arm, stage):
    del self, arm, stage
    return (OptimizerGroupSpec(
        name="synthetic_predictor",
        parameter_names=tuple(name for name, _parameter in model.named_parameters()),
        schedule=ScheduleSpec(peak_lr=0.0003, warmup_start_lr=0.0003,
                              warmup_epochs=0, hold_through_epoch=999,
                              total_epochs=1000, final_lr=0.0003),
        weight_decay=1e-5,
    ),)


@pytest.mark.parametrize("architecture", (baseline.ARM_A, baseline.ARM_L))
def test_original_1000_horizon_restores_exact_epoch101(monkeypatch, tmp_path, architecture):
    monkeypatch.setattr(baseline, "_EPOCH_HORIZON", 1000)
    monkeypatch.setattr(baseline._SyntheticAnchoredProvider, "optimizer_groups", _constant_groups)
    baseline.test_r13_8_workflow_clean_stops_at_100_and_resumes_exactly_to_101(
        tmp_path, architecture,
    )
    state = torch.load(resolve_checkpoint(tmp_path / f"split_{architecture}", "latest"),
                       map_location="cpu", weights_only=False)
    schedule = state["optimizer_schedule_contract"][0]["schedule"]
    assert schedule["total_epochs"] == 1000
    assert schedule["peak_lr"] == schedule["warmup_start_lr"] == schedule["final_lr"] == 0.0003
    assert schedule["warmup_epochs"] == 0 and schedule["hold_through_epoch"] == 999
