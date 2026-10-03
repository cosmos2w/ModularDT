"""Same-weight controls must alter physical assignment and preserve their ledger."""

from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.hypergraph_interventions import (
    fixed_structure_intervention,
    membership_intervention,
)
from honf_forward_core.interface_fields.typed_hypergraph_state import TypedHypergraphState, source_moments


def _state():
    member = torch.tensor([[[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.]]])
    coords = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [99., 99.]]])
    valid = torch.tensor([[True, True, True, False]])
    admission = torch.tensor([[1., 1., 0.]])
    return TypedHypergraphState(memberships={tau: member.clone() for tau in ("MM", "ME", "EM", "QM", "QE")},
        controls={tau: torch.randn(1, 3, 2) for tau in ("MM", "ME", "EM", "QM", "QE")},
        centres=torch.tensor([[[0., 0.], [1., 0.], [100., 100.]]]), admission=admission,
        source_coords={"M": coords, "E": coords}, source_measures={"M": valid.float(), "E": valid.float()},
        source_valid={"M": valid, "E": valid}, source_ids={"M": torch.tensor([[10, 20, 30, -1]]),
                                                               "E": torch.tensor([[40, 50, 60, -1]])},
        strategy_data={"typed_admission": {tau: admission for tau in ("MM", "ME", "EM", "QM", "QE")},
                       "gate_logits": torch.tensor([[2., 1., -1.]])})


@pytest.mark.parametrize("mode", ["geometry", "rewire", "exchange"])
def test_intervention_changes_native_pairs_preserves_positive_density_multiset(mode):
    original = _state()
    altered = membership_intervention(original, mode)
    edge_access = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    baseline = source_moments(edge_access, original.memberships["QM"], original.controls["QM"],
                              original.source_measures["M"], original.source_valid["M"])
    actual = source_moments(edge_access, altered.memberships["QM"], altered.controls["QM"],
                            altered.source_measures["M"], altered.source_valid["M"])
    assert (actual.support != baseline.support).any()
    assert actual.diagnostics["unique_pairs"] == baseline.diagnostics["unique_pairs"]
    for mechanism in original.memberships:
        source, changed = original.memberships[mechanism], altered.memberships[mechanism]
        torch.testing.assert_close(changed.flatten().sort().values, source.flatten().sort().values)
        assert not changed[..., -1].any()  # padded physical source remains ineligible
        torch.testing.assert_close(altered.controls[mechanism], original.controls[mechanism])
        torch.testing.assert_close(altered.source_ids["M"], original.source_ids["M"])
        if mode == "exchange":
            torch.testing.assert_close(changed[:, 2], source[:, 2])


def test_fixed_overlap_changes_admission_without_retraining_memberships():
    original = replace(_state(), admission=torch.tensor([[1., 0., 0.]]))
    changed = fixed_structure_intervention(original, count=2)
    torch.testing.assert_close(changed.admission, torch.tensor([[1., 1., 0.]]))
    for mechanism in original.memberships:
        assert changed.memberships[mechanism] is original.memberships[mechanism]
