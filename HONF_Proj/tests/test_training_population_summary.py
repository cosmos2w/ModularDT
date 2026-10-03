"""Universal-root controls are calibrated only on train inputs and then fixed."""

from dataclasses import replace

import pytest
import torch

from honf_forward_core.interface_fields.adaptive_receiver_hypergraph import AdaptiveReceiverHypergraph
from honf_forward_core.interface_fields.training_population_summary import TrainingPopulationSummaryAccumulator
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _case():
    torch.manual_seed(937)
    return EncodedInterfaceCase(torch.randn(1, 4, 8), torch.randn(1, 5, 8), torch.randn(1, 8),
        torch.rand(1, 4, 2), torch.rand(1, 5, 2), torch.ones(1, 4), torch.randn(1, 4, 3),
        None, torch.ones(1, 5), torch.ones(2))


def test_train_only_calibration_and_fixed_controls_across_held_layouts():
    encoded = _case()
    organizer = AdaptiveReceiverHypergraph(8).eval()
    with torch.no_grad():
        organizer.control_heads["QE"][-1].weight.normal_(0, .2)
    accumulator = TrainingPopulationSummaryAccumulator()
    for offset in (0., .4):
        state = organizer.prepare(encoded, encoded.module_tokens + offset)
        accumulator.add(state, partition="train")
    summary = accumulator.finish()
    held1 = organizer.prepare(encoded, encoded.module_tokens + 2.)
    different = replace(encoded, module_centers=encoded.module_centers + .8,
                         env_coords=encoded.env_coords - .4, module_present=torch.tensor([[1., 1., 1., 0.]]))
    held2 = organizer.prepare(different, different.module_tokens - 1.)
    first, second = summary.apply(held1), summary.apply(held2)
    torch.testing.assert_close(first.controls["QE"], second.controls["QE"])
    assert first.admission.sum() == second.admission.sum() == 1
    assert summary.case_counts[0] == 2
    query = torch.rand(1, 6, 2)
    torch.testing.assert_close(organizer.access(first, query, "QE").weight, torch.ones(1, 6, 5))
    access = organizer.access(second, different.module_centers, "MM")
    assert not access.support[:, 3].any() and not access.support[:, :, 3].any()
    with pytest.raises(ValueError, match="training inputs"):
        accumulator.add(held1, partition="test")
    with pytest.raises(ValueError, match="phase"):
        summary.apply(replace(held1, phase=2))


def test_fixed_summary_changes_actual_trained_reader_and_reports_dense_work():
    encoded = _case()
    model = TypedHypergraphField(8, 12, 2, 2, architecture="adaptive_receiver_hypergraph_honf",
        spatial_dim=2, module_characteristic_length=.1).eval()
    with torch.no_grad():
        model.organizer.geometry_strength["QE"].fill_(8)
        model.organizer.control_heads["QE"][-1].weight.normal_(0, .3)
        model.control_gain["QE"].weight.fill_(.2)
        model.control_score.weight.fill_(.1)
    train = model.organizer.prepare(encoded, encoded.module_tokens)
    accumulator = TrainingPopulationSummaryAccumulator()
    accumulator.add(train, partition="train")
    summary = accumulator.finish()
    held = replace(encoded, env_tokens=encoded.env_tokens + 1.)
    prepared = model.prepare(held, held.module_tokens)
    query, features = torch.rand(1, 7, 2), torch.rand(1, 7, 6)
    normal, _ = model.read(prepared, held, query, features)
    fixed = {**prepared, "hypergraph_plan": summary.apply(prepared["hypergraph_plan"])}
    altered, _ = model.read(fixed, held, query, features)
    assert not torch.allclose(normal, altered)
    baseline_access = model._access(prepared["hypergraph_plan"], query, "QE")
    fixed_access = model._access(fixed["hypergraph_plan"], query, "QE")
    assert fixed_access.support.sum() == 7 * 5
    assert fixed_access.support.sum() > baseline_access.support.sum()
