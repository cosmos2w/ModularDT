"""Runtime activation handling must preserve predictions and training state."""
import copy
from dataclasses import replace

import pytest
import torch

import honf_forward_core.interaction_reused_member_routing.anchored as anchored
from tests.test_e_anchored_packed_execution import fixture


def test_context_restores_default_and_rejects_unknown_mode():
    assert anchored._decoder_execution.get() == "checkpoint"
    with anchored.anchored_decoder_execution("retain"):
        assert anchored._decoder_execution.get() == 1.0
        with anchored.anchored_decoder_execution("checkpoint"):
            assert anchored._decoder_execution.get() == 0.0
        assert anchored._decoder_execution.get() == 1.0
    assert anchored._decoder_execution.get() == "checkpoint"
    with pytest.raises(ValueError):
        with anchored.anchored_decoder_execution("unknown"):
            pass


def test_interleaved_modes_preserve_paired_outputs_gradients_and_adamw(monkeypatch):
    torch.manual_seed(31)
    baseline, scene, query = fixture("local_e_cover")
    query = replace(query, coordinates=query.coordinates.detach(), anchor_weights=query.anchor_weights.detach())
    baseline.query_tile_size = 2
    candidate = copy.deepcopy(baseline)
    calls = []
    checkpoint = anchored.checkpoint

    def counted(*args, **kwargs):
        calls.append(True)
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(anchored, "checkpoint", counted)
    outcomes = []
    for model, modes in ((baseline, ("checkpoint", "checkpoint")),
                         (candidate, ("retain", "checkpoint"))):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        prepared = model.prepare(scene)
        views = []
        for mode in modes:
            with anchored.anchored_decoder_execution(mode):
                full, restricted, _ = model.decode_paired(prepared, query)
                views.extend((full.joint_fields, restricted.joint_fields))
        # Backward occurs after both contexts exit, including the opposite mode.
        sum(view.square().mean() for view in views).backward()
        gradients = {name: None if p.grad is None else p.grad.detach().clone()
                     for name, p in model.named_parameters()}
        optimizer.step()
        outcomes.append(([v.detach().clone() for v in views], gradients,
                         {name: p.detach().clone() for name, p in model.named_parameters()},
                         {name: {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state[p].items()}
                          for name, p in model.named_parameters() if p in optimizer.state}))
    assert len(calls) == 6  # two tiles in each of three checkpointed predictions
    for left, right in zip(outcomes[0][0], outcomes[1][0]):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-11)
    for section in (1, 2):
        for name, left in outcomes[0][section].items():
            right = outcomes[1][section][name]
            if left is None:
                assert right is None
            else:
                torch.testing.assert_close(left, right, rtol=2e-8, atol=2e-9, msg=name)
    assert outcomes[0][3].keys() == outcomes[1][3].keys()
    for name, state in outcomes[0][3].items():
        for key, left in state.items():
            torch.testing.assert_close(left, outcomes[1][3][name][key], rtol=2e-8, atol=2e-9)
