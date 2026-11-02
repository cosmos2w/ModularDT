import torch

from test_relation_anchored_wind import _adapter, _case
from windfarm.inverse.relation_anchored import WindDesignEvaluator


def test_shared_preparation_matches_separate_values_and_design_gradient():
    model, case = _adapter(), _case()
    evaluator = WindDesignEvaluator(model, case)
    coords = torch.tensor([[.6, .7, .9], [1.4, 1.2, 1.1]], dtype=torch.double)
    x = torch.tensor(case.module_centers, dtype=torch.double, requires_grad=True)
    prepared = evaluator.prepare(x)
    shared = torch.cat([evaluator.evaluate_prepared(prepared, part)[0] for part in coords.split(1)], dim=1)
    grad_shared = torch.autograd.grad(shared.square().sum(), x)[0]
    y = x.detach().clone().requires_grad_(True)
    separate = torch.cat([evaluator.evaluate_design(y, part)[0] for part in coords.split(1)], dim=1)
    grad_separate = torch.autograd.grad(separate.square().sum(), y)[0]
    torch.testing.assert_close(shared, separate)
    torch.testing.assert_close(grad_shared, grad_separate)
    assert grad_shared[:, :2].abs().sum() > 0
    assert all(not p.requires_grad for p in model.parameters())
