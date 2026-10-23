"""Scene loss gradients use the complete macro, independently of work VJPs."""
from types import SimpleNamespace

import pytest
import torch

from honf_runtime.joint_native_objective import JointMacroObjective


class _SceneLossCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))

    def joint_begin_update(self, epoch, update_index):
        return {"absolute_epoch": epoch}

    def joint_begin_microbatch(self):
        pass

    def joint_microbatch_result(self):
        zero = self.weight * 0
        constraint = dict(residual_numerator=zero, denominator=len(self.inputs),
                          controller_numerator=zero.detach())
        return {"constraints": {"M": constraint, "E": constraint},
                "additive_losses": {"compact_tie": {
                    "numerator": (self.weight * self.inputs).square().sum(),
                    "denominator": len(self.inputs)}}}

    def joint_constraint_duals(self):
        dual = SimpleNamespace(dual_lambda=torch.tensor(0.), zeta=torch.tensor(0.),
                               penalty_residual=lambda residual: residual * 0)
        return {"M": dual, "E": dual}

    def joint_after_update(self, epoch, update_index, residuals):
        pass


def test_additive_vjp_matches_literal_macro_with_short_final_microbatch():
    core = _SceneLossCore()
    optimizer = torch.optim.SGD(core.parameters(), lr=.01)
    objective = JointMacroObjective(core, optimizer, 4021, 0)
    inputs = torch.tensor([3., 4., 5.], dtype=torch.float64)
    for subset in (inputs[:2], inputs[2:]):
        objective.begin_microbatch()
        core.inputs = subset
        objective.observe(len(subset))
        # The real engine frees each task graph after observing work/scene VJPs.
        (core.weight * subset.sum() / len(inputs)).backward()
    objective.apply()
    expected = inputs.mean() + 2 * core.weight.detach() * inputs.square().mean()
    torch.testing.assert_close(core.weight.grad, expected)
    losses, work = objective.after_update()
    assert losses["compact_tie"] == pytest.approx(float(
        (core.weight.detach() * inputs).square().mean()))
    assert work["compact_tie_case_requests"] == 3
    assert work["joint_case_requests_M"] == 3


def test_additive_loss_rejects_missing_scene_denominator():
    core = _SceneLossCore()
    core.inputs = torch.tensor([3.], dtype=torch.float64)
    original = core.joint_microbatch_result

    def malformed():
        result = original()
        result["additive_losses"]["compact_tie"]["denominator"] = 0
        return result

    core.joint_microbatch_result = malformed
    objective = JointMacroObjective(core, torch.optim.SGD(core.parameters(), lr=.01), 4021, 0)
    with pytest.raises(ValueError, match="positive case count"):
        objective.observe(1)
