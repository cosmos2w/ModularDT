"""Matched native-context controls preserve common exposure and fresh lineage."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from channelthermal.training.campaign import (
    copy_matched_native_context_initial_state,
    heat_null_training_enabled,
    validate_campaign,
    validate_campaign_resume,
)
from channelthermal.training.campaign_response import NativeCampaignResponse

from honf_runtime.config_loader import load_config_bundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from thermal_development import native_context_profiles


def profiles(stage=100):
    manifest = {"manifest_sha256": "a" * 64,
                "partitions": {"train": {"case_ids": ["0348"]}, "test": {"case_ids": ["0277"]}}}
    return native_context_profiles(stage=stage, manifest=manifest)


def effective(profile):
    config = copy.deepcopy(profile)
    config["dataset"] = config["case"]["dataset"]
    config["model"]["channelthermal"] = {"global_feature_schema": "source_local_v3"}
    config["loss"] = {}
    return config


@pytest.mark.parametrize("arm", ["Tree-C", "Global-C"])
def test_native_controls_parse_with_shared_scope_and_explicit_disabled_null(tmp_path, arm):
    initial = profiles()[arm]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(initial))
    bundle = load_config_bundle(path)
    validate_campaign(bundle.effective)
    campaign = initial["training"]["campaign"]
    assert campaign["heat_null_response"] == {"coefficient": 0.0}
    assert not heat_null_training_enabled(campaign)
    assert campaign["physical_loss_policy_version"] == 1
    assert "native_loss_denominators_start_epoch" not in campaign
    assert campaign["response_stencils"][0].endswith("train_0348_responses.npz")
    assert initial["checkpointing"]["save_epoch_milestones"] == list(range(100, 1001, 100))


@pytest.mark.parametrize("arm", ["Tree-C", "Global-C"])
def test_native_e101_amendment_preserves_null_zero_and_existing_structural_calibration(arm):
    initial, continued = effective(profiles()[arm]), effective(profiles(500)[arm])
    checkpoint = {"epoch": 100, "model_config": initial["model"], "train_config": initial,
                  "optimizer_state_dict": {"state": 1}, "rng_state": {"torch": 1}}
    validate_campaign(continued)
    amendment = validate_campaign_resume(checkpoint, continued)
    assert amendment["activation_epoch"] == 101
    assert "structural_measure_policy" not in amendment
    assert amendment["heat_null_response"] == {"coefficient": 0.0}
    checkpoint.update(epoch=500, train_config=copy.deepcopy(continued))
    assert validate_campaign_resume(checkpoint, effective(profiles(1000)[arm])) is None
    checkpoint.update(epoch=99, train_config=initial)
    with pytest.raises(ValueError, match="silently amend"):
        validate_campaign_resume(checkpoint, continued)


@pytest.mark.parametrize("change", ["null_missing", "null_enabled", "gradient", "parent", "horizon", "schema", "structural"])
def test_global_fresh_contract_rejects_unmatched_or_unreviewed_paths(change):
    config = effective(profiles()["Global-C"])
    campaign = config["training"]["campaign"]
    if change == "null_missing":
        campaign.pop("heat_null_response")
    elif change == "null_enabled":
        campaign["heat_null_response"] = {"coefficient": .1}
    elif change == "gradient":
        campaign["organizer_gradient_policy"] = "local_context_shadow_v1"
    elif change == "parent":
        campaign["parent"] = {"checkpoint": "/old.pt"}
    elif change == "horizon":
        campaign["schedule_total_epochs"] = 5000
    elif change == "schema":
        config["model"]["channelthermal"]["global_feature_schema"] = "legacy"
    else:
        campaign["structural_weight"] = .001
    with pytest.raises(ValueError):
        validate_campaign(config)


@pytest.mark.parametrize("arm", ["Tree-C", "Global-C"])
def test_formal_native_recipe_has_distinct_full_data_horizon_and_no_parent(arm):
    config = effective(profiles()[arm])
    config["dataset"] = {}
    config["training"]["campaign"]["schedule_total_epochs"] = 5000
    validate_campaign(config)
    config["training"]["campaign"]["schedule_total_epochs"] = 1000
    with pytest.raises(ValueError, match="Native-context"):
        validate_campaign(config)


def test_explicit_zero_skips_null_construction_even_when_atlas_is_enabled(monkeypatch):
    from channelthermal.interaction_evidence.types import EvidenceSplit
    from channelthermal.training import campaign_null_response, campaign_response

    stencil = SimpleNamespace(split=EvidenceSplit.TRAIN, physical_family_id="train0348")
    monkeypatch.setattr(campaign_response, "load_response_atlas_stencil", lambda path: (stencil, {"anchor_id": "0348"}))
    monkeypatch.setattr(campaign_null_response, "NativeHeatNullResponse",
                        lambda *args, **kwargs: pytest.fail("Disabled null TRAIN must not construct a dataset or wrappers"))
    class Dataset:
        normalizer = SimpleNamespace(stats={})

        def __getitem__(self, index):
            return {}

    dataset = Dataset()
    callback = NativeCampaignResponse(torch.nn.Linear(1, 1), dataset, {},
        {"response_stencils": ["/tmp/train_0348_responses.npz"], "heat_null_response": {"coefficient": 0.0}})
    assert callback.heat_null is None


def fixture_model(extra_tree=False):
    model = torch.nn.Module()
    model.core = torch.nn.Module()
    model.core.context = torch.nn.Linear(2, 2)
    model.core.backend = torch.nn.Module()
    model.core.backend.physical = torch.nn.Linear(2, 2)
    model.core.backend.control_gain = torch.nn.Linear(2, 1)
    model.core.backend.control_score = torch.nn.Linear(2, 1).requires_grad_(False)
    model.core.backend.organizer = torch.nn.Module()
    model.core.backend.organizer.control_geometry_encoder = torch.nn.Linear(2, 2)
    model.core.backend.organizer.control_heads = torch.nn.Linear(2, 2)
    if extra_tree:
        model.core.backend.organizer.tree = torch.nn.Linear(2, 2)
    return model


def test_materialized_copy_matches_context_physical_and_control_heads_but_preserves_tree():
    torch.manual_seed(0)
    canonical = fixture_model()
    target = fixture_model(extra_tree=True)
    independent = target.core.backend.organizer.tree.weight.detach().clone()
    receipt = copy_matched_native_context_initial_state(target, canonical)
    assert receipt["exact_common_tensors_equal"]
    assert all(torch.equal(target.state_dict()[row["name"]], canonical.state_dict()[row["name"]])
               for row in receipt["loaded"])
    assert torch.equal(independent, target.core.backend.organizer.tree.weight)
    assert not target.core.backend.control_score.weight.requires_grad
    target.core.context = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="differently shaped"):
        copy_matched_native_context_initial_state(target, canonical)
