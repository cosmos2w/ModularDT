"""Observation profile dispatch, prefix isolation and saved curve semantics."""
from copy import deepcopy
import json

import numpy as np
import pytest

from honf_runtime.native_prefix_branch import _native_recipe
from honf_runtime.unified_training import _curve_provider_identity, _render_loss_curves
from tests.test_relation_anchored_train_entry import cli
from tests.test_unified_loss_curves import _capture_figures


@pytest.mark.parametrize("dataset,seed_config", [({"name": "collaborator_case"}, {"seed": 91}),
                                                  ("collaborator_case", {"seed": 91}),
                                                  ({"name": "collaborator_case"}, {"initialization": {"seed": 91}})])
def test_entry_dispatches_observation_profile_by_capability_not_case_name(monkeypatch, dataset, seed_config):
    from honf_runtime import observation_response_task

    recipe = {"dataset": dataset, **seed_config, "training_profile": "observation_response",
              "observation_response": {"adapter": "registered_collaborator"}}
    resume = {"epoch": 10, "sampler_state": {"next_epoch": 11}}
    expected = (object(), object())
    calls = []

    def build(actual, *, device, resume_payload):
        calls.append((actual, device, resume_payload))
        return expected

    monkeypatch.setattr(observation_response_task, "build_observation_response_task", build)
    assert cli.build(recipe, device="cpu", resume_payload=resume) == expected
    assert calls == [(recipe, "cpu", resume)]


def test_native_prefix_normalizes_only_the_declared_observation_and_access_changes():
    parent = {"name": "parent", "training_profile": "field_response",
              "member_routing": {"model": {"schema": "joint", "access_measure": "active_domain"}},
              "thermal_field_response": {"variant": "field_only", "native_weight": 1.},
              "training": {"microbatch_cases": 24, "effective_batch_size": 48},
              "dataset": {"name": "the_same_dataset"}, "objective": {"native_weight": 2.}}
    child = deepcopy(parent)
    child.update(name="observation_child", training_profile="observation_response",
                 observation_response={"variant": "spatial_contrasts"})
    child["member_routing"]["model"]["within_role_access_prior"] = "measure_only"
    child['execution_config'] = {'deterministic_algorithms': True}
    original_child = deepcopy(child)
    assert _native_recipe(child) == _native_recipe(parent)
    assert child == original_child
    changed = deepcopy(child)
    changed["objective"]["native_weight"] = 3.
    assert _native_recipe(changed) != _native_recipe(parent)
    changed = deepcopy(child)
    changed["member_routing"]["model"]["within_role_access_prior"] = "unknown"
    assert _native_recipe(changed) != _native_recipe(parent)


def test_observation_deterministic_execution_is_explicit_and_precedes_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.torch, 'use_deterministic_algorithms', lambda value: calls.append(value))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    cli.configure_observation_execution({'training_profile': 'field_response',
                                        'execution_config': {'deterministic_algorithms': True}})
    cli.configure_observation_execution({'training_profile': 'observation_response'})
    assert calls == []
    cli.configure_observation_execution({'training_profile': 'observation_response',
                                        'execution_config': {'deterministic_algorithms': True}})
    assert calls == [True]
    import os
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    with pytest.raises(ValueError, match='must be boolean'):
        cli.configure_observation_execution({'training_profile': 'observation_response',
                                            'execution_config': {'deterministic_algorithms': 'yes'}})


@pytest.mark.parametrize("validation_complete", [False, True])
def test_observation_curve_labels_match_recorded_validation_scope(tmp_path, monkeypatch, validation_complete):
    terms = {"flow/u": 1., "observation_response/extra_absolute": .2,
             "observation_response/point_response": .3, "observation_response/spatial_response": .4}
    validation = {"flow/u": .5}
    if validation_complete:
        validation.update({name: .1 for name in terms if name.startswith("observation_response/")})
    history = [{"epoch": 10, "arm": "R14-Detail-LocalECover", "phase": "joint", "train_losses": terms,
                "validation_objective": {"terms": validation, "total": sum(validation.values())}}]
    metadata = {name: {"panel": "physics", "label": name.split("/")[-1]}
                for name in terms if name.startswith("observation_response/")}
    identity = {"observation_response": {"variant": "spatial_contrasts",
                                           "validation_objective_complete": validation_complete}}
    assert _curve_provider_identity(identity) == identity
    captured = _capture_figures(monkeypatch, tmp_path)
    _render_loss_curves(history, tmp_path, "field_score", loss_metadata=metadata,
                        provider_identity=identity, separate_totals=True)
    assert [item["format"] for item in captured] == ["pdf", "png"]
    for figure in captured:
        assert all(r1 - r0 == 1 and c1 - c0 == 1 for r0, r1, c0, c1 in figure["cells"])
        train = figure["titles"].index("Total TRAIN loss")
        validation_title = "Total measurable validation loss" if validation_complete else "Native-only validation subtotal"
        val = figure["titles"].index(validation_title)
        assert np.allclose(figure["lines"][train][0], [sum(terms.values())])
        assert np.allclose(figure["lines"][val][0], [sum(validation.values())])
        assert figure["labels"][val] == ["VALIDATION: recorded total" if validation_complete
                                         else "VALIDATION: native-only subtotal"]
        assert ("Full validation objective unavailable" in figure["texts"][val]) != validation_complete
    metadata_json = json.loads((tmp_path / "loss_curves_metadata.json").read_text())
    definition = metadata_json["panel_definitions"]["weighted_loss_totals"]
    assert ("native-only subtotal" in definition) != validation_complete
    assert ("all configured shared observation terms" in definition) == validation_complete


@pytest.mark.parametrize("validation_complete", [False, True])
def test_coupled_curve_distinguishes_unavailable_response_validation(tmp_path, monkeypatch, validation_complete):
    history = [{"epoch": 1, "arm": "R14-Detail-LocalECover", "phase": "joint",
                "train_losses": {"flow/u": 1.},
                "validation_objective": {"terms": {"flow/u": .5}, "total": .5}}]
    identity = {"observation_response": {"mode": "coupled_native", "variant": "observed_response",
                                           "validation_objective_complete": validation_complete}}
    captured = _capture_figures(monkeypatch, tmp_path)
    _render_loss_curves(history, tmp_path, "field_score", provider_identity=identity, separate_totals=True)
    for figure in captured:
        title = "Total measurable validation loss" if validation_complete else "Dual-reader native validation subtotal"
        panel = figure["titles"].index(title)
        assert figure["labels"][panel] == ["VALIDATION: dual-reader objective" if validation_complete
                                            else "VALIDATION: dual-reader native subtotal"]
        assert ("Response validation unavailable" in figure["texts"][panel]) != validation_complete
    definition = json.loads((tmp_path / "loss_curves_metadata.json").read_text())["panel_definitions"]["weighted_loss_totals"]
    assert ("response validation measurement is unavailable" in definition) != validation_complete
