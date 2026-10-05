"""Actual tensor control groups stay separate from full-access physical values."""

import json

import torch

from honf_forward_core.evaluation.organization_statistics import TypedOrganizationStatistics
from honf_forward_core.evaluation.typed_work_evidence import TypedWorkEvidenceRecorder
from honf_forward_core.interface_fields.tensor_source_group_residual import TensorSourceGroupResidualField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def test_recorded_tensor_groups_match_execution_and_label_dense_value_plan():
    torch.manual_seed(22)
    encoded = EncodedInterfaceCase(
        torch.rand(1, 2, 8), torch.rand(1, 3, 8), torch.rand(1, 8),
        torch.rand(1, 2, 2), torch.rand(1, 3, 2), torch.ones(1, 2),
        torch.rand(1, 2, 3), None, torch.tensor([[1., 2., 3.]]), torch.ones(1, 1, 2),
        receiver_anchor_coords=torch.rand(1, 4, 2), receiver_anchor_weights=torch.ones(1, 4))
    field = TensorSourceGroupResidualField(8, 12, 2, 2,
        architecture="native_context_global_control_honf", spatial_dim=2,
        module_characteristic_length=.1, options={"global_fast_reader": True, "tensor_source_residual": True}).eval()
    field.set_training_progress(epoch=700)
    query = torch.rand(1, 5, 2)
    features = torch.cat((query, query.square(), query.sin()), -1)
    with torch.no_grad(), TypedOrganizationStatistics(field) as statistics, TypedWorkEvidenceRecorder(field) as recorder:
        state = field.prepare(encoded, encoded.module_tokens)
        field.read(state, encoded, query, features)
    plan = state["hypergraph_plan"].strategy_data["tensor_source_plan"]
    arrays = recorder.arrays
    torch.testing.assert_close(torch.from_numpy(arrays["phase/P0/group_admission"]), plan.admission)
    assert arrays["phase/P0/source_membership/QE"].shape[1] == 3
    assert arrays["phase/P0/value_membership/QE"].shape[1] == 1
    assert arrays["phase/P0/control_presence_scope"].item() == "base Global-C full-access value plan"
    assert arrays["phase/P0/additional_age"].item() == 200
    assert arrays["phase/P0/sparse_fraction"].item() == 1
    provenance = json.loads(arrays["phase/P0/dependency_provenance_json"].item())
    assert provenance["value_donors"] == "all original native fine sources with native eligibility"
    summary = statistics.summary()
    assert summary["phases"]["P0"]["routes"]["QE"]["allocated_group_slots"] == 3
    assert summary["phases"]["P0"]["routes"]["QE"]["active_frontier_groups"] == int((plan.admission > 0).sum())
    assert summary["native_routes"]["P0/QE"]["native_support_fraction"] == 1
