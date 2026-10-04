"""Guard the direct inverse reader against silently expanding a development cohort."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import thermal_campaign_heat_inference as inverse


@pytest.fixture
def reader(monkeypatch):
    import thermal_development
    from channelthermal.data import datasets
    from channelthermal.evaluation import loading

    selected = {"manifest_sha256": "a" * 64,
                "partitions": {"test": {"case_ids": ["0601", "0602"]}}}
    checkpoint = {"epoch": 100, "train_config": {"dataset": {}},
                  "global_normalization_stats": {"field_mean_by_channel": [2.0] * 5,
                                                  "field_std_by_channel": [3.0] * 5}}
    seen = []

    def construct(path, **kwargs):
        seen.append(kwargs)
        # A selected validation reader cannot safely fall back to packed stats.
        assert kwargs["case_ids"] == ("0601", "0602")
        assert np.array_equal(kwargs["normalizer"].stats["field_mean_by_channel"], [2.0] * 5)
        return SimpleNamespace(selected_case_ids=["0601", "0602"])

    monkeypatch.setattr(loading, "load_model", lambda *args: (torch.nn.Module(), checkpoint))
    monkeypatch.setattr(datasets, "GlobalChannelThermalDataset", construct)
    monkeypatch.setattr(thermal_development, "resolve_evaluation_manifest", lambda *args, **kwargs: selected)
    monkeypatch.setattr(inverse, "screen_indices", lambda *args: [])
    return checkpoint, seen


def test_direct_inverse_reader_preserves_saved_train_normalizer_and_test_membership(reader, tmp_path):
    _checkpoint, seen = reader
    result = inverse.evaluate_heat(tmp_path / "checkpoint.pt", dataset_path=tmp_path / "data.h5",
                                   output_dir=tmp_path / "evaluation", evaluation_scope="development")
    assert result.is_dir() and len(seen) == 1
    assert seen[0]["split"] == "test"


@pytest.mark.parametrize("scope", ["auto", "formal-full"])
def test_inverse_dev_reader_rejects_missing_normalization_before_case_access(reader, tmp_path, monkeypatch, scope):
    import thermal_development

    checkpoint, seen = reader
    checkpoint["global_normalization_stats"] = {}
    if scope == "formal-full":
        checkpoint["train_config"]["dataset"]["development_manifest"] = "fixed.json"
        monkeypatch.setattr(thermal_development, "resolve_evaluation_manifest", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="saved selected-training normalization"):
        inverse.evaluate_heat(tmp_path / "checkpoint.pt", dataset_path=tmp_path / "data.h5",
                              output_dir=tmp_path / "evaluation", evaluation_scope=scope)
    assert not seen and not (tmp_path / "evaluation").exists()


def test_inverse_evidence_resume_cannot_reuse_another_membership(reader, tmp_path):
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "summary.json").write_text(json.dumps({
        "checkpoint": str(tmp_path / "checkpoint.pt"), "checkpoint_epoch": 100,
        "development_manifest_sha256": "b" * 64, "evaluation_scope": "development"}))
    with pytest.raises(ValueError, match="changed checkpoint or evaluation budget"):
        inverse.evaluate_heat(tmp_path / "checkpoint.pt", dataset_path=tmp_path / "data.h5",
                              output_dir=output, evaluation_scope="development", resume=True)


def test_inverse_cli_exposes_deliberate_full_scope(tmp_path):
    args = inverse.parse_args(["--checkpoint", "checkpoint.pt", "--dataset", "data.h5",
                               "--output-dir", str(tmp_path), "--evaluation-scope", "formal-full"])
    assert args.evaluation_scope == "formal-full" and args.development_manifest is None
