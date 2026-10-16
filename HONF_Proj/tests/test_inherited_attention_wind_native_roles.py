from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "src", PROJECT / "Case_WindFarm/src"):
    sys.path.insert(0, str(directory))

from windfarm.training import inherited_attention_task as iag_wind
from windfarm.training.content_hypergraph_task import (
    ROLE_NAMES,
    WindContentHypergraphTask,
    WindContentTargets,
    WindFormalA2DTaskV1,
)

ROLE_COUNTS = (1640, 1640, 1640, 1640, 1632)
ROLE_IDS = np.repeat(np.arange(5, dtype=np.int8), ROLE_COUNTS)


def _empty_provider() -> iag_wind.WindInheritedAttentionTaskV1:
    return object.__new__(iag_wind.WindInheritedAttentionTaskV1)


def test_native_role_batches_use_fixed_q8192_and_preserve_formal_monitor_api(monkeypatch: pytest.MonkeyPatch):
    provider = _empty_provider()
    monkeypatch.setattr(provider, "native_evaluation_rows", lambda split, case_keys: (17, 23))
    calls = []

    def batch_builder(received_provider, row, *, split, batch_index):
        calls.append((received_provider, row, split, batch_index))
        return SimpleNamespace(
            case_keys=(row,),
            receivers=SimpleNamespace(role_ids=ROLE_IDS[None, :]),
            auxiliary={"role_query_counts": dict(zip(ROLE_NAMES, ROLE_COUNTS))},
        )

    monkeypatch.setattr(iag_wind, "make_native_role_panel_batch", batch_builder)
    validation = tuple(provider.native_role_evaluation_batches("validation"))
    train = tuple(provider.native_role_evaluation_batches("train"))

    assert [batch.case_keys for batch in validation] == [(17,), (23,)]
    assert [batch.case_keys for batch in train] == [(17,), (23,)]
    assert [(call[1], call[2], call[3]) for call in calls] == [
        (17, "validation", 0), (23, "validation", 1),
        (17, "train-fixture", 0), (23, "train-fixture", 1),
    ]
    assert WindFormalA2DTaskV1.VALIDATION_QUERY_COUNT == 10240
    assert iag_wind.WindInheritedAttentionTaskV1.validation_batches is WindFormalA2DTaskV1.validation_batches
    assert iag_wind.WindInheritedAttentionTaskV1.native_evaluation_batches is WindContentHypergraphTask.native_evaluation_batches


def test_native_role_metrics_keep_hash_and_physical_five_role_tails(monkeypatch: pytest.MonkeyPatch):
    provider = _empty_provider()
    metadata = {"row_index": 17, "layout_index": 4, "n_turbines": 8}
    targets = WindContentTargets(
        np.zeros((1, 8192, 3), dtype=np.float32), (17,), (metadata,),
    )
    predictions = SimpleNamespace(
        main_mps=torch.ones((1, 8192, 3)),
        role_ids=torch.as_tensor(ROLE_IDS[None, :]),
    )
    query_hash = "a" * 64
    record = provider.native_role_evaluation_metrics(
        predictions,
        targets,
        {"query_sampling_sha256": query_hash},
    )
    monkeypatch.setattr(provider, "native_evaluation_rows", lambda split, case_keys: (17,))
    metrics = provider.reduce_native_role_evaluation_metrics([record], split="validation")

    assert metrics["query_count_per_row"] == 8192
    assert metrics["role_query_counts"] == dict(zip(ROLE_NAMES, ROLE_COUNTS))
    assert metrics["row_query_sampling_sha256"] == {"17": query_hash}
    assert metrics["overall"]["roles"][ROLE_NAMES[0]]["vector_rmse_mps"]["mean"] == pytest.approx(np.sqrt(3.0))
    assert metrics["overall"]["roles"][ROLE_NAMES[-1]]["component_rmse_mps"]["Ux"]["count"] == 1


def test_native_role_metrics_reject_bad_sampler_receipt_or_row_inventory(monkeypatch: pytest.MonkeyPatch):
    provider = _empty_provider()
    metadata = {"row_index": 17, "layout_index": 4, "n_turbines": 8}
    targets = WindContentTargets(np.zeros((1, 8192, 3), dtype=np.float32), (17,), (metadata,))
    predictions = SimpleNamespace(
        main_mps=torch.zeros((1, 8192, 3)),
        role_ids=torch.as_tensor(ROLE_IDS[None, :]),
    )
    with pytest.raises(ValueError, match="query SHA256"):
        provider.native_role_evaluation_metrics(predictions, targets, {"query_sampling_sha256": "bad"})
    record = provider.native_role_evaluation_metrics(
        predictions, targets, {"query_sampling_sha256": "b" * 64},
    )
    monkeypatch.setattr(provider, "native_evaluation_rows", lambda split, case_keys: (23,))
    with pytest.raises(ValueError, match="exact selected rows"):
        provider.reduce_native_role_evaluation_metrics([record], split="validation")
