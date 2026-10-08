from __future__ import annotations

import numpy as np
import pytest
from windfarm.workflows.evaluate_forward import (
    _default_evaluation_root,
    _select_evaluation_rows,
)


def test_default_evaluation_root_handles_canonical_and_legacy_checkpoints(tmp_path):
    for checkpoint in (tmp_path / "latest_model.pt", tmp_path / "checkpoints/latest_model.pt"):
        assert _default_evaluation_root(checkpoint, "validation") == tmp_path / "evaluations/validation"


def test_explicit_evaluation_rows_preserve_requested_order_and_split_membership() -> None:
    split_rows = np.asarray([3, 7, 11, 19], dtype=np.int64)

    selected = _select_evaluation_rows(split_rows, [19, 7])

    np.testing.assert_array_equal(selected, [19, 7])


@pytest.mark.parametrize("requested", ([7, 7], [], [5]))
def test_explicit_evaluation_rows_reject_duplicates_empty_or_out_of_split(requested) -> None:
    with pytest.raises(ValueError, match="row-indices"):
        _select_evaluation_rows([3, 7, 11], requested)


def test_evaluation_rows_default_to_full_split() -> None:
    np.testing.assert_array_equal(_select_evaluation_rows([3, 7, 11], None), [3, 7, 11])
