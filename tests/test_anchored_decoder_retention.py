"""Selective tile retention preserves paired outputs and live gradients."""
import copy

import pytest
import torch

import honf_forward_core.interaction_reused_member_routing.anchored as anchored
from tests.test_e_anchored_packed_execution import fixture


def test_even_selection_counts_partial_tile():
    assert anchored.retained_decoder_tile_indices(0, .5) == ()
    assert anchored.retained_decoder_tile_indices(5, .5) == (0, 2, 4)
    assert anchored.retained_decoder_tile_indices(5, 0) == ()
    assert anchored.retained_decoder_tile_indices(5, 1) == (0, 1, 2, 3, 4)
    for invalid in (-.1, 1.1, float('nan'), True):
        with pytest.raises(ValueError):
            anchored.retained_decoder_tile_indices(5, invalid)


def test_retention_preserves_paired_results_and_backward_outside_scope(monkeypatch):
    reference, scene, query = fixture('local_e_cover')
    candidate = copy.deepcopy(reference)
    reference.query_tile_size = candidate.query_tile_size = 3
    calls = []
    original = anchored.checkpoint

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(anchored, 'checkpoint', record)
    results = []
    for model, fraction in ((reference, 0), (candidate, .5)):
        before = len(calls)
        with anchored.anchored_decoder_execution(fraction):
            full, restricted, counts = model.decode_paired(model.prepare(scene), query)
        assert len(calls) - before == (2 if fraction == 0 else 1)
        loss = full.joint_fields.square().sum() + restricted.joint_fields.square().sum()
        gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        results.append((full.joint_fields, restricted.joint_fields, counts, gradients))
    for actual, expected in zip(results[1][:2], results[0][:2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert results[1][2] == results[0][2]
    for actual, expected in zip(results[1][3], results[0][3]):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_execution_scope_restores_default_on_exception():
    with pytest.raises(RuntimeError):
        with anchored.anchored_decoder_execution('retain'):
            assert anchored._decoder_execution.get() == 1
            raise RuntimeError('stop')
    assert anchored._decoder_execution.get() == 'checkpoint'


def test_provider_scope_does_not_retain_separate_auxiliary_decode():
    from tools.relation_anchored_train import configure_native_retention

    class Provider:
        def predict_native(self):
            return anchored._decoder_execution.get()

    provider = Provider()
    configure_native_retention(provider, .5)
    assert provider.predict_native() == .5
    assert anchored._decoder_execution.get() == 'checkpoint'
