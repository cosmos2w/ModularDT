"""Transparent phase recording and actual physical pair intervention audits."""

import numpy as np
import pytest
import torch
from torch import nn

from honf_forward_core.evaluation.typed_work_evidence import TypedWorkEvidenceRecorder, compare_native_access
from honf_forward_core.interface_fields.typed_hypergraph_state import TypedHypergraphState, TypedSourceAccess


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def _access(self, plan, receivers, mechanism):
        self.calls += 1
        weight = torch.ones(1, receivers.shape[1], 2)
        return TypedSourceAccess(
            weight, weight, torch.zeros(*weight.shape, 16), weight.bool(), torch.ones(1, receivers.shape[1], 1)
        )

    @staticmethod
    def _ledger(mechanism, access, work):
        return {f"hypergraph_{mechanism}_executed_rows": torch.as_tensor(work)}

    def prepare(self, phase):
        source = {kind: torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]) for kind in ("M", "E")}
        plan = TypedHypergraphState(
            {"MM": torch.ones(1, 1, 2)},
            {"MM": torch.zeros(1, 1, 16)},
            torch.zeros(1, 1, 2),
            torch.ones(1, 1),
            source,
            {kind: torch.ones(1, 2) for kind in source},
            {kind: torch.ones(1, 2, dtype=torch.bool) for kind in source},
            {kind: torch.tensor([[0, 1]]) for kind in source},
            phase=phase,
        )
        access = self._access(plan, source["M"], "MM")
        self._ledger("MM", access, 4)
        return {"hypergraph_plan": plan}


def test_recorder_keeps_all_three_phases_actual_receivers_and_original_results():
    backend = Backend().eval()
    assert "prepare" not in backend.__dict__ and "_access" not in backend.__dict__
    with TypedWorkEvidenceRecorder(backend) as record:
        for phase in range(3):
            state = backend.prepare(phase)
            result = backend._access(state["hypergraph_plan"], torch.tensor([[[0.2, 0.3]]]), "QM")
            torch.testing.assert_close(result.weight, torch.ones(1, 1, 2))
    assert backend.calls == 6
    assert "prepare" not in backend.__dict__ and "_access" not in backend.__dict__
    for phase in range(3):
        assert f"phase/P{phase}/source_membership/MM" in record.arrays
        assert int(record.arrays[f"access/P{phase}/MM/00000/executor/hypergraph_MM_executed_rows"]) == 4
        np.testing.assert_array_equal(
            record.arrays[f"access/P{phase}/QM/00000/receivers"], np.asarray([[[0.2, 0.3]]], dtype=np.float32)
        )
    assert all(row["changed_pairs"] == 0 for row in compare_native_access(record.arrays, record.arrays).values())
    assert "_ledger" not in backend.__dict__


def test_recorder_restores_methods_on_exception_and_rejects_training():
    backend = Backend()
    with pytest.raises(ValueError, match="evaluation-only"):
        TypedWorkEvidenceRecorder(backend)
    backend.eval()
    with pytest.raises(RuntimeError), TypedWorkEvidenceRecorder(backend):
        raise RuntimeError("probe interrupted")
    assert "prepare" not in backend.__dict__ and "_access" not in backend.__dict__


def test_native_pair_audit_checks_identity_geometry_cardinality_and_weight_multisets():
    backend = Backend().eval()
    with TypedWorkEvidenceRecorder(backend) as record:
        backend.prepare(0)
    before = record.arrays
    after = {key: value.copy() for key, value in before.items()}
    prefix = "access/P0/MM/00000"
    before[f"{prefix}/weight"][:] = [[[1.0, 0.0], [0.0, 2.0]]]
    before[f"{prefix}/support"][:] = before[f"{prefix}/weight"] > 0
    after[f"{prefix}/weight"][:] = [[[0.0, 1.0], [2.0, 0.0]]]
    after[f"{prefix}/support"][:] = after[f"{prefix}/weight"] > 0
    audit = compare_native_access(before, after)["P0/MM"]
    assert audit["changed_pairs"] == 4
    assert audit["normal_positive_pairs"] == audit["intervention_positive_pairs"] == 2
    assert audit["per_receiver_weight_multiset_equal"] and audit["per_receiver_support_cardinality_equal"]
    after[f"{prefix}/source_coords"][0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="geometry"):
        compare_native_access(before, after)
