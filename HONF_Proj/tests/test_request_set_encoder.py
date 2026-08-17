"""Request `R` set invariance with context `c`; no `D`, `G`, or `G_hat`."""

from __future__ import annotations

import torch

from honf_inverse_core.models.request_encoder import RequestSetEncoder
from tests.inverse_test_utils import permute_request, request_batch


def test_request_encoder_is_permutation_invariant_and_context_sensitive() -> None:
    torch.manual_seed(1)
    model = RequestSetEncoder(hidden_dim=32, layers=2, dropout=0.0).eval()
    request = request_batch(2)
    context = torch.zeros(2, 10)
    geometry = torch.zeros(2, 8)
    first = model(request, context, geometry)
    order = torch.tensor([1, 0, 3, 2])
    second = model(permute_request(request, order), context, geometry)
    torch.testing.assert_close(first.global_embedding, second.global_embedding)
    changed_context = context.clone()
    changed_context[:, 0] = 2.0
    changed = model(request, changed_context, geometry)
    assert not torch.allclose(first.global_embedding, changed.global_embedding)
    assert first.token_embeddings.shape == (2, 4, 32)
