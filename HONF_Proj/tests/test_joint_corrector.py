"""One bounded correction over planned `G`, design `D`, and realized `G_hat`."""

from __future__ import annotations

import torch

from honf_inverse_core.models.joint_corrector import JointConsistencyCorrector


def test_joint_corrector_is_single_pass_and_bounded() -> None:
    torch.manual_seed(4)
    model = JointConsistencyCorrector(
        num_edges=3, max_modules=5, condition_dim=16, hidden_dim=32,
        blocks=2, dropout=0.0, max_plan_delta=0.03, max_layout_delta=0.04,
    )
    plan = torch.rand(2, 3, 12)
    plan[..., 0] = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    realized = torch.rand(2, 3, 12)
    layout = torch.rand(2, 5, 3)
    present = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)
    result = model(plan, layout, present, realized, torch.zeros(2, 4), torch.zeros(2, 16))
    assert result.delta_plan.abs().max() <= 0.030001
    assert result.delta_layout.abs().max() <= 0.040001
    assert torch.all(result.delta_layout[present == 0] == 0)
    assert result.magnitude.shape == (2,)
    torch.testing.assert_close(
        result.corrected_plan[..., 0].sum(dim=1), plan[..., 0].sum(dim=1)
    )
