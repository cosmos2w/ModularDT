from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from honf_runtime.unified_training import (
    FRESH_MATCHED_NATIVE_VALIDATION_POLICY,
    EngineConfig,
    SamplingKey,
    TaskBatch,
    TrainingEngine,
    _apply_fresh_native_validation_policy,
    _validate_fresh_matched_initialization,
    fresh_matched_initialization_receipt,
    fresh_matched_recipe_seed,
)

_HONF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HONF_ROOT / "tools"))
import joint_regional_train as fresh_runner


def _confirmation_seed_recipe(tmp_path, task="wind", preferred="P-R"):
    decision = {
        "schema": "conditional_second_seed_eligibility_v1",
        "chosen_dataset": task, "preferred_mode": preferred,
        "second_seed": 43 if task == "wind" else 1,
        "physical_direction_pass": True, "robustness_pass": True,
        "mandatory_measurements_complete": True, "unresolved_implementation_failure": False,
        "q8192_cost_ratio": 1.25, "mandatory_measurements_sha256": "a" * 64,
        "wind_eligible": task == "wind",
        "primary_endpoints": [
            {"task": domain, "mode": mode, "epoch": 2500, "checkpoint_sha256": "b" * 64}
            for domain in ("thermal", "wind") for mode in ("P", "P-R", "P-R2")
        ],
    }
    path = tmp_path / "eligibility.json"
    payload = json.dumps(decision).encode()
    path.write_bytes(payload)
    recipe = {"task": task, "mode": preferred, "seed": decision["second_seed"],
              "conditional_confirmation": {
                  "schema": "conditional_second_seed_multiview_v1", "task": task,
                  "seed": decision["second_seed"], "preferred_mode": preferred,
                  "decision_path": str(path), "decision_sha256": hashlib.sha256(payload).hexdigest(),
              }}
    return recipe, decision, path


@pytest.mark.parametrize("task,seed", [("thermal", 0), ("wind", 42)])
def test_primary_seed_guard_cannot_silently_become_confirmation(task, seed):
    assert fresh_matched_recipe_seed({"task": task, "seed": seed}) == seed
    with pytest.raises(ValueError, match="fixed dataset seed"):
        fresh_matched_recipe_seed({"task": task, "seed": seed + 1})
    with pytest.raises(ValueError, match="Conditional confirmation"):
        fresh_matched_recipe_seed({"task": task, "seed": seed, "conditional_confirmation": None})


@pytest.mark.parametrize("task", ["thermal", "wind"])
def test_confirmation_pair_requires_completed_frozen_decision(tmp_path, task):
    recipe, _, _ = _confirmation_seed_recipe(tmp_path, task)
    assert fresh_matched_recipe_seed(recipe, verify_decision=True) == recipe["seed"]
    recipe["mode"] = "P"
    assert fresh_matched_recipe_seed(recipe, verify_decision=True) == recipe["seed"]
    recipe["mode"] = "P-R2"
    with pytest.raises(ValueError, match="one preferred-routing/P pair"):
        fresh_matched_recipe_seed(recipe, verify_decision=True)


@pytest.mark.parametrize("mutation", ["missing_endpoint", "unfinished", "no_cost", "damage", "thermal_over_wind"])
def test_confirmation_rejects_partial_or_ineligible_primary_evidence(tmp_path, mutation):
    recipe, decision, path = _confirmation_seed_recipe(tmp_path, "thermal")
    if mutation == "missing_endpoint":
        decision["primary_endpoints"].pop()
    elif mutation == "unfinished":
        decision["primary_endpoints"][0]["epoch"] = 1500
    elif mutation == "no_cost":
        decision["mandatory_measurements_complete"] = False
    elif mutation == "damage":
        decision["robustness_pass"] = False
    else:
        decision["wind_eligible"] = True
    payload = json.dumps(decision).encode()
    path.write_bytes(payload)
    recipe["conditional_confirmation"]["decision_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="all six endpoints"):
        fresh_matched_recipe_seed(recipe, verify_decision=True)


class _SamplerProvider:
    def __init__(self):
        self.keys = []

    def identity_payload(self):
        return {"task": "Synthetic", "manifest_fingerprint": "a" * 64}

    def epoch_cases(self, epoch, seed):
        return tuple(f"case-{index}" for index in range(9))

    def make_batch(self, case_keys, key):
        self.keys.append(key)
        return TaskBatch(None, None, None, case_keys=tuple(case_keys))


def test_training_engine_exposes_fit_identical_epoch_order_and_update_microbatches():
    config = EngineConfig(
        seed=42,
        microbatch_cases=2,
        effective_cases=5,
        total_epochs=10,
        training_mode="joint",
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
    )
    engine = TrainingEngine(config, device="cpu")
    provider = _SamplerProvider()
    cases, dataset_id = engine.epoch_case_order(provider, 1)
    repeated, repeated_dataset_id = engine.epoch_case_order(provider, 1)
    assert cases == repeated
    assert len(cases) == 9
    assert dataset_id == repeated_dataset_id
    slices = engine.epoch_update_slices(len(cases))
    assert [cases[part] for part in slices] == [cases[:5], cases[5:]]

    first = engine.prepare_update_microbatches(
        provider,
        arm="P-R2",
        phase="joint",
        epoch=1,
        update_index=0,
        update_cases=cases[slices[0]],
        sampling_dataset_id=dataset_id,
    )
    second = engine.prepare_update_microbatches(
        provider,
        arm="P-R2",
        phase="joint",
        epoch=1,
        update_index=1,
        update_cases=cases[slices[1]],
        sampling_dataset_id=dataset_id,
    )
    assert [batch.case_keys for batch in first] == [tuple(cases[:2]), tuple(cases[2:4]), tuple(cases[4:5])]
    assert [batch.case_keys for batch in second] == [tuple(cases[5:7]), tuple(cases[7:9])]
    assert len(provider.keys) == 5
    assert {key.sampling_version for key in provider.keys} == {SamplingKey.CASE_EPOCH_VERSION}
    assert {key.dataset_id for key in provider.keys} == {dataset_id}
    assert [(key.update_index, key.microbatch_index) for key in provider.keys] == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
    ]


def test_fresh_build_dispatches_thermal_dev_materialization_flag(monkeypatch):
    from types import SimpleNamespace

    from channelthermal.training import joint_task as thermal_joint_task

    seen = {}

    def fake_factory(**kwargs):
        seen.update(kwargs)
        provider = SimpleNamespace(identity_payload=lambda: {"task": "ThermalChannel"})
        return nn.Linear(1, 1), provider

    monkeypatch.setattr(thermal_joint_task, "build_thermal_joint_task", fake_factory)
    recipe = {
        "task": "thermal",
        "mode": "P",
        "seed": 0,
        "hidden": 1,
        "message": 1,
        "regional_anchors": 0,
        "depth": 2,
        "receiver_tile": 512,
        "primary_queries": 1,
        "microbatch_cases": 1,
        "effective_cases": 1,
        "formal_full": False,
        "total_epochs": 2,
    }
    _, _, engine, _ = fresh_runner.build(
        recipe, device="cpu", include_dev_targets=False, require_clean_source=False,
    )
    assert seen["include_dev_targets"] is False
    assert engine.config.curve_every == 10


@pytest.mark.parametrize("task", ("thermal", "wind"))
@pytest.mark.parametrize("dev_materialized", (False, True))
def test_fresh_prepare_requests_and_enforces_no_dev_targets(
    monkeypatch, tmp_path, task, dev_materialized,
):
    from types import SimpleNamespace

    manifest_path = tmp_path / "fresh-manifest.json"
    manifest = {
        "task": task,
        "arms": {
            mode: {
                "recipe_path": str(tmp_path / f"{mode}.json"),
                "run_id": f"{mode}-fresh-run",
                "recipe_sha256": "a" * 64,
                "output_dir": str(tmp_path / f"output-{mode}"),
            }
            for mode in ("P", "P-R", "P-R2")
        },
    }
    monkeypatch.setattr(
        fresh_runner, "_read_fresh_comparison_manifest",
        lambda *_args, **_kwargs: (manifest, manifest_path),
    )
    monkeypatch.setattr(
        fresh_runner, "read_recipe",
        lambda path: {"task": task, "mode": Path(path).stem},
    )
    monkeypatch.setattr(fresh_runner, "_bind_calibration_receipt", lambda recipe: recipe)
    seen = []

    class StopAfterPreparationCheck(Exception):
        pass

    def fake_build(recipe, **kwargs):
        seen.append(kwargs.get("include_dev_targets"))
        provider = SimpleNamespace()
        if task == "thermal":
            provider.dev_targets_materialized = dev_materialized
            provider.validation_cases = ("DEV-0",) if dev_materialized else ()
        else:
            provider._validation_cache = object() if dev_materialized else None
        return object(), provider, None, {}

    monkeypatch.setattr(fresh_runner, "build", fake_build)
    monkeypatch.setattr(
        fresh_runner, "summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StopAfterPreparationCheck()),
    )
    if dev_materialized:
        with pytest.raises(RuntimeError, match="preparation must keep all DEV targets|lazy DEV validation cache"):
            fresh_runner.prepare_fresh_matched_comparison(manifest_path)
    else:
        with pytest.raises(StopAfterPreparationCheck):
            fresh_runner.prepare_fresh_matched_comparison(manifest_path)
    assert seen == [False]


@pytest.mark.parametrize("task", ("thermal", "wind"))
def test_fresh_matched_allocator_seals_three_collision_checked_recipe_ids(tmp_path, task):
    config, config_path, config_sha256 = fresh_runner.fresh_matched_config(task)
    assert config["engine_schedule"]["curve_every"] == 10
    campaign_root = tmp_path / f"{task}-campaign"
    manifest = fresh_runner.allocate_fresh_matched_comparison(task, campaign_root)
    assert manifest["source_config_path"] == str(config_path)
    assert manifest["source_config_sha256"] == config_sha256
    assert manifest["optimizer_updates"] == 0
    assert manifest["optimizer_state_created"] is False
    assert set(manifest["arms"]) == {"P", "P-R", "P-R2"}
    prefix = "TMV1-" if task == "thermal" else "WMV1-"
    run_ids = []
    for mode, arm in manifest["arms"].items():
        run_id = arm["run_id"]
        assert run_id.startswith(prefix) and len(run_id) == len(prefix) + 32
        assert int(run_id[len(prefix):], 16) >= 0
        assert not Path(arm["output_dir"]).exists()
        run_ids.append(run_id)
        recipe = fresh_runner.read_recipe(arm["recipe_path"])
        assert recipe["seed"] == config["seed"]
        assert recipe["engine_schedule"]["curve_every"] == 10
        if mode == "P":
            fresh_runner._validate_fresh_matched_recipe(recipe)
            legacy = copy.deepcopy(recipe)
            legacy["engine_schedule"]["curve_every"] = 100
            fresh_runner._validate_fresh_matched_recipe(legacy)
            for unsupported in (5, 20, True):
                invalid = copy.deepcopy(recipe)
                invalid["engine_schedule"]["curve_every"] = unsupported
                with pytest.raises(ValueError, match="engine_schedule"):
                    fresh_runner._validate_fresh_matched_recipe(invalid)
            invalid_other = copy.deepcopy(recipe)
            invalid_other["engine_schedule"]["monitor_every"] = 50
            with pytest.raises(ValueError, match="engine_schedule"):
                fresh_runner._validate_fresh_matched_recipe(invalid_other)
        assert recipe["initialization"] == "fresh_all_trainable"
        assert recipe["native_validation_policy"] == FRESH_MATCHED_NATIVE_VALIDATION_POLICY
        if mode == "P-R2":
            assert recipe["model_contract"]["native_routing"] == {
                "schema": "native_interaction_routing_v2",
                "mode": "P-R2",
                "policy": "learned_density",
                "views": 2,
                "feature_blocks": [64, 64],
                "score_hidden": 32,
                "score_bound": 2.0,
            }
        if task == "thermal":
            assert recipe["flow_readout_law"] == "native_curl_cell_centred_v1"
            assert recipe["calibration_receipt"]["required_payload_sha256"] == (
                "sealed_after_fresh_P_native_curl_calibration")
    assert len(run_ids) == len(set(run_ids))


def test_fresh_source_seal_adds_r2_scorer_only_to_fresh_identity():
    legacy = fresh_runner.training_source_identity(require_clean=False)
    fresh = fresh_runner.training_source_identity(
        require_clean=False,
        flow_readout_law="native_curl_cell_centred_v1",
        fresh_matched=True,
    )
    scorer_path = "HONF_Proj/src/honf_forward_core/interface_fields/native_interaction_routing.py"
    assert scorer_path not in legacy["source_sha256"]
    assert scorer_path in fresh["source_sha256"]
    assert "HONF_Proj/src/config_core/forward/joint_regional/fresh_multiview_thermal_v1.json" in fresh[
        "source_sha256"]
    assert "HONF_Proj/Case_ThermalChannel/src/channelthermal/flow_curl.py" in fresh["source_sha256"]


class _ValidationCore:
    def __init__(self):
        self.consolidated_native_validation = False

    def set_consolidated_native_validation(self, enabled):
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        self.consolidated_native_validation = enabled


class _ValidationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.25]))
        self.core = _ValidationCore()


def test_fresh_validation_policy_receipt_is_strict_and_reapplied_after_state_load():
    provider_identity = {
        "fresh_matched_comparison": True,
        "task": "ThermalChannel",
        "manifest_fingerprint": "b" * 64,
        "model_config": {"mode": "P-R2", "routing_schema": "native_interaction_routing_v2"},
        "normalization_stats_sha256": "c" * 64,
        "response_scales": {"fluid": 1.0},
        "loss_weights": {"flow": 0.5, "temperature": 0.5},
        "auxiliary_calibration": {"response": 0.2, "operator": 0.3},
    }
    model = _ValidationModel()
    receipt = fresh_matched_initialization_receipt(
        model,
        provider_identity,
        active_mode="P-R2",
        seed=42,
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        deterministic_case_order=True,
    )
    assert receipt["native_validation_policy"] == FRESH_MATCHED_NATIVE_VALIDATION_POLICY
    _validate_fresh_matched_initialization(
        receipt,
        model,
        provider_identity,
        arm="P-R2",
        seed=42,
        sampling_version=SamplingKey.CASE_EPOCH_VERSION,
        deterministic_case_order=True,
    )
    drifted = dict(receipt, native_validation_policy="legacy_eager")
    with pytest.raises(ValueError, match="native-validation policy"):
        _validate_fresh_matched_initialization(
            drifted,
            model,
            provider_identity,
            arm="P-R2",
            seed=42,
            sampling_version=SamplingKey.CASE_EPOCH_VERSION,
            deterministic_case_order=True,
        )

    restored = _ValidationModel()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.core.consolidated_native_validation is False
    _apply_fresh_native_validation_policy(restored)
    assert restored.core.consolidated_native_validation is True
    assert "consolidated_native_validation" not in restored.state_dict()
