"""CPU integration checks against the fixed TRAIN panel and response atlas."""
from __future__ import annotations

import copy
import pickle
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch

from honf_runtime.contextual_frontier_training import load_contextual_frontier_config
from honf_runtime.thermal_soft_organization import select_calibration_panel
from honf_runtime.unified_training import SamplingKey, capture_rng_state
from channelthermal.training.contextual_joint_sharing import functional_probe
from channelthermal.training.contextual_joint_task import build_thermal_contextual_frontier_task
from channelthermal.training.unified_task import DEFAULT_ATLAS_DIRECTORY


_PROJECT = Path(__file__).resolve().parents[1]
_DATA = Path("/data/wanglz/ModularDT/1_ChannelThermal/Processed_ChannelThermal_Dataset/packed_dataset.h5")
_MANIFEST = Path("/data/wanglz/ModularDT/thermal_development/fixed25_v1/manifest.json")


def _assert_tree_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, Mapping):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def _contains_tensor(value):
    if torch.is_tensor(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


@pytest.fixture(scope="module")
def contextual_tasks():
    if not _DATA.is_file() or not _MANIFEST.is_file() or not DEFAULT_ATLAS_DIRECTORY.is_dir():
        pytest.skip("The fixed Thermal TRAIN data and existing 0348 response atlas are unavailable.")
    torch.set_num_threads(1)
    tasks = {}
    for arm, slug in (("J-H", "jh"), ("J-D", "jd")):
        config, _ = load_contextual_frontier_config(
            _PROJECT / f"configs/research/thermal_contextual_frontier_{slug}_dev_v1.json"
        )
        model, provider = build_thermal_contextual_frontier_task(config, arm=arm, device="cpu")
        provider.primary_queries = 16
        tasks[arm] = (model, provider)
    return tasks


def _sampling_key(provider, model, *, epoch=21, update=0):
    return SamplingKey(
        provider.seed,
        epoch,
        update,
        0,
        "joint",
        model.architecture,
        sampling_version=provider.sampling_version,
        dataset_id=provider.sampling_dataset_id,
    )


def _first_multi_module_panel_position(provider):
    panel = select_calibration_panel(
        provider.training_cases, limit=min(96, len(provider.training_cases))
    )
    positions = {case_id: index for index, case_id in enumerate(provider.training_case_ids)}
    panel = tuple(
        case_id for case_id in panel
        if int((np.asarray(provider.training_cases[positions[case_id]]["structure"]["module_present"]) > 0.5).sum()) > 1
    )
    return panel, 0


def test_both_arms_execute_matched_coarse_and_fine_subtree_probes(contextual_tasks):
    for arm, (model, provider) in contextual_tasks.items():
        panel, position = _first_multi_module_panel_position(provider)
        terms, detail = functional_probe(provider, 2 * position, 21)
        assert detail["case_id"] == panel[position]
        assert detail["coarse_K"] < detail["fine_K"]
        assert detail["ancestry_ids"]
        assert detail["encoder_reused_for_intervention"] is True
        assert detail["native_forward_calls"] == 3
        assert set(terms) == {"share_module", "risk_module"}
        assert all(row["gradients"] for row in terms.values())


def test_native_control_atlas_calibration_and_resume_are_exact(contextual_tasks):
    model, provider = contextual_tasks["J-H"]
    assert provider.responses.family_ids == ("0348",)
    assert provider.responses.heat_pairs
    assert provider.responses.heat_scales["fluid_fields/u"] < 1e-3
    assert provider.responses.heat_scales["fluid_fields/v"] < 1e-3
    assert provider.responses.heat_scales["fluid_fields/p"] < 1e-3

    pair = provider.responses.heat_pairs[0]
    case_id = "0348"
    assert case_id in provider.training_case_ids
    batch = provider.make_batch((case_id,), _sampling_key(provider, model))
    structure = batch.scene_inputs.structure
    predictions, _ = provider.predict_native(
        model, batch.scene_inputs, batch.receivers, model.architecture, "joint", epoch=21
    )
    native = predictions.native_prepared
    scene = native.scene_batch
    mask = structure["module_present"] > 0.5
    torch.testing.assert_close(scene.module_mask, mask)
    torch.testing.assert_close(scene.module_ids, structure["module_source_ids"])
    torch.testing.assert_close(
        scene.module_features[..., -1], batch.receivers.heat / model.heat_scale
    )
    active_scene_ids = tuple(str(int(v)) for v in structure["module_source_ids"][0, mask[0]])
    atlas_slot_ids = tuple(
        int(str(value).rsplit(":", 1)[-1]) for value in pair.baseline.design.active_module_ids
    )
    assert active_scene_ids == tuple(map(str, atlas_slot_ids))
    assert len(active_scene_ids) == len(pair.baseline.design.active_module_ids)

    response_loss, response_detail = provider.responses.loss(model, 0)
    assert torch.isfinite(response_loss)
    assert response_detail["kind"] == "heat"

    # Exercise the native-loss callback before backward; its raw organizer VJPs
    # must coexist with the caller's unchanged native graph.
    model.train()
    train_case = _first_multi_module_panel_position(provider)[0][1]
    train_batch = provider.make_batch((train_case,), _sampling_key(provider, model))
    train_predictions, _ = provider.predict_native(
        model, train_batch.scene_inputs, train_batch.receivers, model.architecture, "joint", epoch=21
    )
    loss_terms = provider.loss_terms(train_predictions, train_batch.targets, "joint", None)
    native_loss = sum(term.numerator / term.denominator * term.weight for term in loss_terms.values())
    native_loss.backward()
    assert provider.organizer.last_collection["banks"]["module"]["eligible_cases"] == 1
    assert provider.organizer.last_collection["banks"]["environment"]["eligible_cases"] == 1
    assert provider.organizer._accumulators["capacity_module"]["gradients"]
    assert provider.organizer._accumulators["reinforce_module"]["gradients"]

    # Keep this real-data calibration short while covering every input-only
    # module-count stratum: one case per observed stratum, small query budget.
    counts = {
        int((np.asarray(case["structure"]["module_present"]) > 0.5).sum())
        for case in provider.training_cases
    }
    provider.organizer.calibration_max_cases = len(counts)
    provider.microbatch_size = min(4, len(counts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first_parameter = next(model.parameters())
    optimizer.state[first_parameter] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.full_like(first_parameter, 0.125),
        "exp_avg_sq": torch.full_like(first_parameter, 0.25),
    }

    model_state = copy.deepcopy(model.state_dict())
    provider_counters = (provider.training_samples, provider.validation_samples)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    clock = (int(model.core.organization_epoch), int(model.core.organization_update))
    rng = capture_rng_state()
    report = provider.organizer.calibrate(
        provider, 21, optimizer=optimizer, panel_runner=provider._calibration_batches
    )
    assert report["optimizer_steps"] == 0
    assert report["native_cases_collected"] == len(counts)
    assert report["model_provider_optimizer_rng_restored"] is True
    assert report["functional_probe_exposure"]
    assert len(report["functional_probe_exposure"]) == 4
    assert Counter(row["bank"] for row in report["functional_probe_exposure"]) == {
        "module": 2, "environment": 2,
    }
    assert all("skipped" not in row for row in report["functional_probe_exposure"])
    for name in ("share_module", "risk_module", "share_environment", "risk_environment"):
        assert report["terms"][name]["coefficient"] > 0.0
    _assert_tree_equal(model_state, model.state_dict())
    _assert_tree_equal(optimizer_state, optimizer.state_dict())
    _assert_tree_equal(
        gradients,
        {name: None if parameter.grad is None else parameter.grad for name, parameter in model.named_parameters()},
    )
    assert provider_counters == (provider.training_samples, provider.validation_samples)
    assert clock == (int(model.core.organization_epoch), int(model.core.organization_update))
    _assert_tree_equal(rng, capture_rng_state())

    losses = {"native": float(native_loss.detach())}
    work = {}
    provider.before_optimizer_step(model, optimizer, losses, work, epoch=22, update_index=0)
    assert "contextual_native_full_gradient_norm" in work
    assert "contextual_organizer_gradient_norm" in work
    assert provider.organizer.state["last_update"]["organizer_weighted_gradient_norm"] <= (
        0.1 * provider.organizer.state["last_update"]["native_full_gradient_norm"] + 1e-8
    )

    saved_state = provider.training_state_dict()
    encoded_state = pickle.dumps(saved_state, protocol=pickle.HIGHEST_PROTOCOL)
    assert len(encoded_state) < 2 * 1024 * 1024
    assert not _contains_tensor(saved_state)
    saved_samples = provider.training_samples
    provider.training_samples = -1
    provider.load_training_state_dict(saved_state)
    assert provider.training_samples == saved_samples
    _assert_tree_equal(saved_state, provider.training_state_dict())

    restored = type(model).from_state_payload(
        model.state_payload(), state_dict=copy.deepcopy(model.state_dict()), device="cpu"
    )
    restored.core.set_organization_clock(21, 0)
    replay = restored.prepare_native(
        batch.scene_inputs.structure,
        batch.receivers.fluid_xy,
        physical_heat=batch.receivers.heat,
        local_query_points=batch.receivers.local_query_points,
        ntheta=batch.receivers.ntheta,
        scene_key=batch.scene_inputs.case_ids,
    )
    original_identity = native.organization.module_banks[0].frontier.sampling_identity
    replay_identity = replay.organization.module_banks[0].frontier.sampling_identity
    assert replay_identity == original_identity


def test_primary_schedule_is_matched_across_joint_arms(contextual_tasks):
    batches=[]
    for arm,(model,provider) in contextual_tasks.items():
        batches.append(provider.make_batch(("0348",),_sampling_key(provider,model,epoch=100,update=2)))
    a,b=batches
    assert a.case_keys==b.case_keys
    assert a.auxiliary["query_sampling_sha256"]==b.auxiliary["query_sampling_sha256"]
    torch.testing.assert_close(a.receivers.fluid_xy,b.receivers.fluid_xy,rtol=0,atol=0)
    torch.testing.assert_close(a.receivers.local_query_points,b.receivers.local_query_points,rtol=0,atol=0)
    torch.testing.assert_close(a.targets.field_targets,b.targets.field_targets,rtol=0,atol=0,equal_nan=True)
