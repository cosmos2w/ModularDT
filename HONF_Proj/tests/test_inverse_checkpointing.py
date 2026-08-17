"""Provenance-complete checkpoint round-trip for `R,c -> G -> D -> G_hat`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from honf_inverse_core.models.hierarchical_inverse import HierarchicalInverseDesigner
from honf_inverse_core.training.checkpointing import load_inverse_checkpoint, save_inverse_checkpoint


def _designer() -> HierarchicalInverseDesigner:
    return HierarchicalInverseDesigner.from_config(
        {
            "num_edges": 3, "max_modules": 5, "request_hidden_dim": 16,
            "plan_hidden_dim": 24, "plan_layers": 1, "layout_hidden_dim": 24,
            "layout_layers": 1, "corrector_enabled": False, "dropout": 0.0,
        }
    )


def _provenance() -> dict:
    return {
        "forward_checkpoint_id": "best_predicted_model.pt",
        "inverse_dataset_version": 1,
        "inverse_dataset_hash": "abc",
        "request_schema_version": 1,
        "compact_plan_schema_version": 1,
        "normalization_stats": {"mean": [0.0]},
    }


def test_inverse_checkpoint_round_trip_and_required_provenance(tmp_path: Path) -> None:
    destination = save_inverse_checkpoint(
        tmp_path / "best_plan_model.pt",
        designer=_designer(), stage="stage_plan", epoch=2, global_step=7,
        provenance=_provenance(),
    )
    checkpoint = load_inverse_checkpoint(destination)
    assert checkpoint["stage"] == "stage_plan"
    assert checkpoint["global_step"] == 7
    loaded = HierarchicalInverseDesigner.load(destination)
    assert loaded.plan_flow.num_edges == 3
    bad = _provenance()
    bad.pop("inverse_dataset_hash")
    with pytest.raises(ValueError, match="missing"):
        save_inverse_checkpoint(
            tmp_path / "bad.pt", designer=_designer(), stage="stage_plan",
            epoch=0, global_step=0, provenance=bad,
        )


def test_pre_ordered_plan_checkpoint_has_exact_zero_compatibility_path() -> None:
    source = _designer()
    legacy = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith("layout_flow.ordered_plan_projection.")
    }
    restored = _designer()
    restored.load_compatible_state_dict(legacy)
    assert torch.count_nonzero(restored.layout_flow.ordered_plan_projection.weight) == 0
    assert torch.count_nonzero(restored.layout_flow.ordered_plan_projection.bias) == 0


def test_public_loader_rejects_noninverse_and_incomplete_checkpoints(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.pt"
    torch.save({"model_config": {}, "model_state_dict": {}}, wrong)
    with pytest.raises(ValueError, match="schema mismatch"):
        HierarchicalInverseDesigner.load(wrong)

    incomplete = tmp_path / "incomplete.pt"
    torch.save(
        {
            "checkpoint_schema_name": "honf_hierarchical_inverse",
            "checkpoint_schema_version": 1,
            "model_config": _designer().model_config,
            "model_state_dict": _designer().state_dict(),
            "provenance": {"forward_checkpoint_id": "best_predicted_model.pt"},
        },
        incomplete,
    )
    with pytest.raises(ValueError, match="provenance is incomplete"):
        HierarchicalInverseDesigner.load(incomplete)
