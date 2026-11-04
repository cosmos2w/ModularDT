from types import SimpleNamespace

import pytest
import torch

from honf_forward_core.interaction_reused_member_routing.anchored import AnchoredMemberRoutingCore


def carrier(dim, slots):
    scene = SimpleNamespace(coordinate_dim=dim, num_environment_tokens=slots, batch_size=1,
        environment_mask=torch.ones((1, slots), dtype=torch.bool), environment_measure=torch.ones((1, slots)))
    query = SimpleNamespace(batch_size=1, num_queries=1, coordinates=torch.zeros((1, 1, dim)),
        anchor_indices=torch.arange(slots).reshape(1, 1, slots),
        anchor_weights=torch.full((1, 1, slots), 1 / slots), query_mask=torch.ones((1, 1), dtype=torch.bool))
    return scene, query


def test_cover_bound_tracks_dimension_and_validates_weights():
    core = SimpleNamespace(locator_mode='local_e_cover', _query_mask=lambda query: query.query_mask)
    validate = AnchoredMemberRoutingCore._validate_anchor_carrier
    scene, query = carrier(3, 8)
    assert validate(core, scene, query)[0].shape[-1] == 8
    for dim, slots in ((3, 9), (2, 5)):
        with pytest.raises(ValueError, match='tensor-product cover'):
            validate(core, *carrier(dim, slots))
    query.anchor_weights *= .5
    with pytest.raises(ValueError, match='sum to one'):
        validate(core, scene, query)
