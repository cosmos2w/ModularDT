from types import SimpleNamespace

import pytest
import torch
from channelthermal.training.joint_task import JointThermalTask


def _context(present, groups):
    batch = present.shape[0]
    return SimpleNamespace(
        present=present,
        environment_coords=torch.zeros(batch, 4, 2),
        group_states=torch.zeros(batch, groups, 8) if groups else None,
    )


def _native(context, *, labels, affine, read_width=None, unique_counts=None):
    batch = context.present.shape[0]
    native = SimpleNamespace(
        context=context,
        flow_receivers=torch.zeros(batch, labels, 2),
        grid_indices=torch.zeros(batch, affine, 2, dtype=torch.long),
    )
    if unique_counts is not None:
        # Three label rows intentionally repeat one primary cell in each case.
        # Their true primary-cell counts are 2, while the five-point unions have
        # 4 and 5 entries. Work accounting must not call duplicate labels new
        # neighbour reads.
        gather = torch.zeros(batch, labels, 5, dtype=torch.long)
        if labels == 3:
            gather[0, :, 0] = torch.tensor([0, 0, 1])
            if batch > 1:
                gather[1, :, 0] = torch.tensor([0, 1, 1])
        else:
            gather[:, :, 0] = torch.arange(labels).expand(batch, -1)
        native.native_curl_stencil = SimpleNamespace(
            unique_counts=torch.tensor(unique_counts), gather_indices=gather,
        )
        native.flow_query_receivers = torch.zeros(batch, read_width, 2)
    return native


@pytest.mark.parametrize("mode,groups", [("P", 0), ("P-G", 2), ("P-H", 2)])
def test_increment_response_adds_affine_reads_but_no_flow_reads_and_counts_duplicate_label_curl_union(mode, groups):
    provider = object.__new__(JointThermalTask)
    provider.joint_mode = mode
    provider.operator_rows_per_case = 128
    provider.use_operator = True

    main_context = _context(torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), groups=groups)
    response_context = _context(torch.tensor([[1.0, 1.0, 1.0]]), groups=groups)
    main = _native(main_context, labels=3, affine=2, read_width=5, unique_counts=[4, 5])
    # The increment receiver tensor is intentionally large. apply_native(...,
    # increment=True) does not invoke nonlinear flow prediction for it.
    response = _native(response_context, labels=7, affine=4)
    predictions = SimpleNamespace(
        work={},
        native_prepared=main,
        response_prepared=response,
    )
    batch = SimpleNamespace(targets=SimpleNamespace(case_ids=("a", "b"), case_indices=(0, 1)))

    counts = JointThermalTask.work_counts(provider, batch, predictions, {"work": {}})

    assert counts["pair_contexts_prepared"] == 2
    assert counts["native_primary_label_receivers"] == 6
    assert counts["native_primary_unique_receivers"] == 4
    assert counts["native_flow_logical_unique_receivers"] == 9
    assert counts["native_curl_neighbor_added_unique_receivers"] == 5
    assert counts["native_curl_union_padding_receiver_slots"] == 1
    assert counts["native_flow_receivers"] == 10  # B * padded union width, main context only
    assert counts["source_conditioned_flow_read_executed_pairs"] == 30
    assert counts["source_conditioned_flow_read_active_pairs"] == 13
    assert counts["active_physical_source_receiver_pairs"] == 13
    assert counts["response_affine_union_read_executed_pairs"] == 12
    assert counts["response_increment_flow_read_executed_receivers"] == 0
    assert counts["response_increment_flow_read_executed_pairs"] == 0
    assert counts["collective_receiver_access_score_slots"] == (0 if groups == 0 else 36)
    if groups:
        assert counts["collective_receiver_access_score_slots"] != 36 + 1 * 7 * groups
