from __future__ import annotations

import torch

from windfarm.workflows.independent_direct_pair import (
    DirectPairSplitProvenance,
    aggregate_observed_g2_actions,
)


def test_overlapping_g2_actions_keep_unknown_pairs_unknown() -> None:
    reached_nodes = torch.tensor([[True, True, False]])
    targets = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    observed = torch.tensor(
        [
            [True, True, False, True],
            [False, True, False, True],
            [False, False, False, False],
        ]
    )
    eligible = torch.ones((1, 4), dtype=torch.bool)

    target_pairs, observed_pairs = aggregate_observed_g2_actions(
        reached_nodes,
        targets,
        observed,
        eligible,
    )

    assert torch.equal(observed_pairs, torch.tensor([[False, True, False, True]]))
    assert torch.equal(target_pairs, torch.tensor([[False, True, False, False]]))


def test_selected_comparison_row_layout_is_held_out_from_direct_fit() -> None:
    split = {
        "split_sha256": "frozen-split",
        "training_layout_indices": [10, 20, 30],
        "development_layout_indices": [40],
        "training_layouts": [
            {"layout_index": 10, "rows_direction_order": [1, 2]},
            {"layout_index": 20, "rows_direction_order": [3, 4]},
            {"layout_index": 30, "rows_direction_order": [5, 6]},
        ],
        "development_layouts": [{"layout_index": 40, "rows_direction_order": [7, 8]}],
    }

    provenance = DirectPairSplitProvenance.from_frozen_g2_split(
        split,
        g2_search_report_sha256="g2-report",
        g2_label_checkpoint_sha256="g2-checkpoint",
        student_checkpoint_sha256="student-checkpoint",
        comparison_row_index=4,
    )

    assert provenance.heldout_training_layout_index == 20
    assert provenance.fit_layout_indices == (10, 30)
    assert provenance.fit_row_indices == (1, 2, 5, 6)
    assert provenance.heldout_row_indices == (3, 4)
    assert provenance.comparison_row_index == 4
    assert provenance.development_rows_excluded == (7, 8)
    assert provenance.as_dict()["development_g2_labels_loaded"] is False
