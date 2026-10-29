"""Evidence reductions must distinguish live source mass from actual reads."""
import numpy as np
import pytest
import torch

from honf_runtime.r13_12_mediated_evaluation import (
    QUERY_IDS, TRAIN24_CASES, _flatten_arrays, summarize_mediated_trace,
)


def test_empty_route_mass_and_two_restrictions_use_actual_composition():
    trace = {
        "anchor_read_indices": torch.tensor([[0, 0, 0], [0, 1, 0]]),
        "anchor_read_E_slots": torch.tensor([0, 1]),
        "anchor_read_member_read_index": torch.tensor([0, 0, 0, 1, 1]),
        "anchor_read_member_index": torch.tensor([0, 2, 3, 1, 3]),
        "anchor_read_access_prior": torch.tensor([.5, .4, .2, .7, .1]),
        "anchor_read_probabilities": torch.tensor([1., .75, .25, 1., 1.]),
        "anchor_read_strength": torch.tensor([[.4, .6], [0., .8]]),
        "mediated_source_alpha": torch.tensor([[.2, .5, .3], [.1, .2, .7]]),
        "mediated_source_selected": torch.tensor([[True, True, False], [True, False, True]]),
        "mediated_environment_prior": torch.tensor([[.4, .3], [0., .1]]),
        "mediated_conditional_rows": [(torch.tensor([0, 1]), torch.tensor([[.6, .4], [0., 0.]]), torch.tensor([4., 0.]))],
    }
    result = summarize_mediated_trace(trace, module_count=2, full_environment_mass=np.array([.7, .3]))
    assert result["source_mass_assigned_to_empty_ME_rows"]["mean"] == pytest.approx(.5)
    assert result["omitted_E_path_mass_due_to_source_masking"]["mean"] == pytest.approx(.1)
    assert result["retained_E_path_mass_relative_to_full_composition"]["mean"] == pytest.approx((.6 / .7 + .1 / .3) / 2)
    assert result["zero_M_source_fraction"] == 0
    assert result["zero_M_value_strength_fraction"] == .5
    assert result["final_own_E_beta_before_locator_blend"]["mean"] == pytest.approx((.75 * .6 + .8) / 2)


def test_ragged_trace_export_preserves_ids_without_pickle_objects():
    arrays, metadata = {}, {}
    _flatten_arrays([(torch.tensor([37, 11]), torch.tensor([[0., 1.], [1., 0.]])), "cycle1/me"],
                    "conditional", arrays, metadata)
    np.testing.assert_array_equal(arrays["conditional_0_0"], [37, 11])
    assert metadata["conditional_1"] == "cycle1/me"
    assert all(array.dtype.kind != "O" for array in arrays.values())
    assert QUERY_IDS.shape == (512,) and len(np.unique(QUERY_IDS)) == 512
    assert len(TRAIN24_CASES) == len(set(TRAIN24_CASES)) == 24
