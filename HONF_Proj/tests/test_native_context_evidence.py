"""Native instrumentation preserves the operator and method/hook ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_forward_core.evaluation.native_context_evidence import NativeContextEvidence
from honf_forward_core.interface_fields.common import SharedInterfaceContext


def test_native_context_records_executed_paths_and_restores_normal_exception_exits():
    torch.manual_seed(4)
    common = SharedInterfaceContext(hidden_dim=8, field_dim=2, num_heads=2,
        coarse_latent_count=3, coarse_blocks=1, local_radius_factor=1., fourier_frequencies=2).eval()
    states, environment = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
    present, measure = torch.ones(1, 2), torch.tensor([[.3, .7]])
    centres = torch.tensor([[[0., 0.], [1., 0.]]])
    receivers = torch.tensor([[[.1, 0.], [4., 0.]]], requires_grad=True)
    features, receiver_features = torch.randn(1, 2, 3), torch.randn(1, 2, 4)
    global_token, scale = torch.randn(1, 8), torch.ones(1, 1, 2)
    encoded = SimpleNamespace(module_centers=centres, env_coords=torch.rand(1, 2, 2),
                              module_present=present, env_weights=measure)

    def prepare():
        return SimpleNamespace(encoded=encoded,
            coarse_state=common.prepare_coarse(states, environment, present, measure))

    core = SimpleNamespace(training=False, common=common, prepare=prepare)

    def run():
        prepared = core.prepare()
        coarse = common.read_coarse(receiver_features, global_token, prepared.coarse_state)
        local, _ = common.read_local(receivers, states, centres, features, present, scale, .5)
        return coarse + local

    expected = run()  # normal lazy materialization precedes instrumentation
    expected_gradient = torch.autograd.grad(expected.square().sum(), receivers)[0]
    original_prepare = core.prepare
    handle = common.local_message.register_forward_pre_hook(lambda *_args: None)
    hook_keys = tuple(common.local_message._forward_pre_hooks)
    with NativeContextEvidence(core, save_arrays=True) as evidence:
        actual = run()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), receivers)[0],
                               expected_gradient, rtol=0, atol=0)
    row = evidence.summary()["phases"]["P0"]
    assert row["coarse_source_attention_cells"] == 24
    assert row["coarse_self_attention_cells"] == 18
    assert row["coarse_receiver_attention_cells"] == 12
    assert row["local_executed_message_rows"] == 1
    assert row["local_allocated_geometry_pairs"] == 4
    np.testing.assert_array_equal(evidence.arrays["P0/read0000/local_support"], [[[True, False], [False, False]]])
    kernel = (.8 ** 2) * 1.4
    assert evidence.arrays["P0/read0000/local_kernel_weight"][0, 0, 0] == pytest.approx(kernel / (1 + kernel))
    assert core.prepare is original_prepare
    assert "read_local" not in common.__dict__
    with pytest.raises(RuntimeError, match="intentional"), NativeContextEvidence(core):
        run()
        raise RuntimeError("intentional")
    assert core.prepare is original_prepare
    assert "prepare_coarse" not in common.__dict__
    assert tuple(common.local_message._forward_pre_hooks) == hook_keys
    torch.testing.assert_close(run(), expected, rtol=0, atol=0)
    handle.remove()
