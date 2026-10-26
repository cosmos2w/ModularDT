from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
for _directory in (PROJECT / "src", PROJECT / "Case_ThermalChannel/src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from channelthermal.training.member_routing_health import (
    MemberRoutingHealth, evaluate_pause, request_clean_stop, _native_with_first_tile_trace,
)


def _case(case_id, count, *, m_support=0, e_support=0, gradient=0.0, sensitivity=0.0, noise=0.0):
    return {
        "case_id": case_id,
        "module_count": count,
        "support": {
            "final_M": {"positive_pairs": m_support},
            "final_E": {"positive_pairs": e_support},
        },
        "live_gradient_norms": {
            "proposal_score_M": gradient,
            "score_M": gradient,
            "key_M": gradient,
            "value_M": gradient,
            "B_M": gradient,
            "score_E": gradient,
            "key_E": gradient,
            "value_E": gradient,
            "B_E": gradient,
        },
        "typed_read_heat_sensitivity_l2": {"M": sensitivity, "E": sensitivity},
        "typed_read_repeatability_l2": {"M": noise, "E": noise},
        "typed_read_strength_l2": {"M": 0.0, "E": 0.0},
    }


def _summary(cases):
    paths = {}
    for path, key_names in (
        ("M", ("proposal_score_M", "score_M", "key_M", "value_M", "B_M")),
        ("E", ("proposal_score_M", "score_E", "key_E", "value_E", "B_E")),
    ):
        positive = sum(case["support"][f"final_{path}"]["positive_pairs"] for case in cases)
        norms = {name: max((case["live_gradient_norms"][name] for case in cases), default=0.0)
                 for name in key_names}
        paths[path] = {
            "positive_weight_pairs": positive,
            "active_in_any_query": positive > 0,
            "semantic_gradient_norms_max_across_panel": norms,
            "semantic_gradients_exact_zero_across_panel": all(value == 0.0 for value in norms.values()),
        }
    return {"cases": cases, "paths": paths}


def test_pause_on_previously_active_m_path_collapse_even_when_e_remains_live():
    initial = _summary([
        _case("0001", 1, m_support=1, e_support=1, gradient=0.5, sensitivity=0.2),
        _case("0012", 12, m_support=2, e_support=1, gradient=0.5, sensitivity=0.2),
    ])
    current = _summary([
        _case("0001", 1, e_support=1), _case("0002", 2),
        _case("0005", 5), _case("0012", 12, e_support=2),
    ])

    decision = evaluate_pause(initial, {"M": True, "E": True}, current, (1, 2, 5, 12))

    assert decision == {
        "kind": "previously_active_typed_path_collapsed",
        "path": "M",
        "panel_module_counts": [1, 2, 5, 12],
    }


def test_one_empty_case_or_query_does_not_trigger_widespread_collapse():
    current = _summary([
        _case("0001", 1), _case("0002", 2, m_support=1, gradient=0.2),
        _case("0005", 5, m_support=1, gradient=0.2),
        _case("0012", 12, m_support=1, gradient=0.2),
    ])

    assert evaluate_pause(None, {"M": True, "E": True}, current, (1, 2, 5, 12)) is None


def test_native_output_insensitivity_pauses_even_when_other_task_gradients_are_live():
    roles = ("temperature/fluid", "temperature/surface", "temperature/material")
    def observation(signal):
        cases = [_case(str(count), count, m_support=1, e_support=1, gradient=.1)
                 for count in (1, 2, 5, 12)]
        for case in cases:
            case["heat_sensitivity"] = {
                "heat_plus10_fieldwise_native_delta": dict.fromkeys(roles, signal),
                "repeatability_fieldwise_native_delta": dict.fromkeys(roles, 1e-7),
            }
        return _summary(cases)
    initial, current = observation(.02), observation(0.)
    decision = evaluate_pause(initial, {"M": True, "E": True}, current, (1, 2, 5, 12))
    assert decision["kind"] == "native_temperature_heat_response_became_unresolved"
    current["cases"][-1]["heat_sensitivity"]["heat_plus10_fieldwise_native_delta"][roles[0]] = .01
    assert evaluate_pause(initial, {"M": True, "E": True}, current, (1, 2, 5, 12)) is None


def test_native_health_trace_tiles_full_observation_union_and_keeps_task_gradients():
    from types import SimpleNamespace
    from honf_forward_core.content_hypergraph.contracts import QueryBatch

    calls = []
    parameter = torch.nn.Parameter(torch.tensor(2.))
    class Core:
        query_tile_size = 4
        def decode(self, organization, query, *, return_trace):
            calls.append((query.num_queries, return_trace))
            scores = parameter * query.coordinates[..., :1]
            fields = scores.expand(-1, -1, 4)
            return SimpleNamespace(joint_fields=fields, work_counts={"rows": query.num_queries},
                                   trace={"proposal_logits_M": scores} if return_trace else None)
    model = SimpleNamespace(core=Core(), _extract_native_roles=lambda prepared, temperature, flow:
                            {"temperature": temperature, "flow": flow})
    prepared = SimpleNamespace(organization=object(), query_batch=QueryBatch(
        coordinates=torch.ones(1, 11, 2)))
    native, counts, trace = _native_with_first_tile_trace(model, prepared)
    assert calls == [(4, True), (4, False), (3, False)]
    assert counts == {"rows": 11}
    assert native["temperature"].shape == (1, 11, 1)
    score_gradient, parameter_gradient = torch.autograd.grad(
        native["temperature"].sum(), (trace["proposal_logits_M"], parameter))
    assert torch.equal(score_gradient, torch.ones_like(score_gradient))
    assert parameter_gradient == 11


def test_pause_when_both_paths_lose_initial_native_signal_beyond_repeatability():
    initial = _summary([
        _case("0001", 1, m_support=1, e_support=1, gradient=0.5, sensitivity=0.2),
        _case("0012", 12, m_support=1, e_support=1, gradient=0.5, sensitivity=0.2),
    ])
    current = _summary([
        _case("0001", 1, noise=1e-7), _case("0002", 2, noise=1e-7),
        _case("0005", 5, noise=1e-7), _case("0012", 12, noise=1e-7),
    ])

    decision = evaluate_pause(initial, {"M": False, "E": False}, current, (1, 2, 5, 12))

    assert decision is not None
    assert decision["kind"] == "both_typed_paths_lost_native_gradient_and_scene_sensitivity"


def _health_fixture():
    class Provider:
        device = "cpu"
        seed = 0
        primary_queries = 1024
        sampling_version = "case_epoch_v1"
        sampling_dataset_id = "fixed-train"
        training_samples = 12
        validation_samples = 3

        def __init__(self):
            self.training_cases = []
            for count, case_id in ((1, "0001"), (2, "0002"), (5, "0005"), (12, "0012")):
                present = np.zeros(12, dtype=np.float32)
                present[:count] = 1
                self.training_cases.append({"case_id": case_id, "structure": {"module_present": present}})
            self.manifest = {"partitions": {"train": {"case_ids": [case["case_id"] for case in self.training_cases]}}}

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._joint_epoch = 4
            self._joint_update_index = 5
            self._joint_macro_clock = 6

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Core()
            self.layer = torch.nn.Linear(1, 1)
            self.architecture = "R13-3-Route"

    provider = Provider()
    model = Model()
    model.train()
    model.layer.eval()  # Check that mixed submodule flags are restored exactly.
    model.layer.weight.grad = torch.full_like(model.layer.weight, 3.0)
    health = MemberRoutingHealth(provider, {
        "enabled": True, "observation_epochs": 12, "later_every": 10,
        "panel_module_counts": [1, 2, 5, 12], "pause_on_collapse": True,
    })
    return provider, model, health


def test_observation_restores_rng_modes_counters_clocks_and_existing_gradients():
    provider, model, health = _health_fixture()
    parameter = model.layer.weight
    original_grad = parameter.grad
    grad_value = original_grad.clone()
    flags = tuple(module.training for module in model.modules())
    clocks = (model.core._joint_epoch, model.core._joint_update_index, model.core._joint_macro_clock)
    counters = (provider.training_samples, provider.validation_samples)
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()

    def fake_run_case(self, observed_model, case_id):
        random.random()
        np.random.random()
        torch.rand(1)
        self.provider.training_samples += 1
        observed_model.core._joint_epoch = 99
        observed_model.core._joint_update_index = 98
        observed_model.core._joint_macro_clock = 97
        observed_model.train()
        parameter.grad = torch.zeros_like(parameter)
        case = next(case for case in self.provider.training_cases if case["case_id"] == case_id)
        count = int(np.asarray(case["structure"]["module_present"]).sum())
        return _case(case_id, count, m_support=1, e_support=1, gradient=0.1,
                     sensitivity=0.2, noise=0.01)

    health._run_case = MethodType(fake_run_case, health)
    observation = health.observe(model, training_epoch=1)

    assert observation["schema"] == "thermal_member_routing_health_v1"
    assert tuple(module.training for module in model.modules()) == flags
    assert (provider.training_samples, provider.validation_samples) == counters
    assert (model.core._joint_epoch, model.core._joint_update_index, model.core._joint_macro_clock) == clocks
    assert parameter.grad is original_grad
    torch.testing.assert_close(parameter.grad, grad_value, rtol=0, atol=0)
    assert random.getstate() == py_rng
    after_np = np.random.get_state()
    assert after_np[0] == np_rng[0]
    np.testing.assert_array_equal(after_np[1], np_rng[1])
    assert after_np[2:] == np_rng[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, rtol=0, atol=0)


def test_health_state_is_compact_and_clean_stop_never_replaces_an_existing_request(tmp_path):
    provider, _model, health = _health_fixture()
    initial = _summary([
        _case("0001", 1, m_support=1, e_support=1),
        _case("0002", 2, m_support=1, e_support=1),
        _case("0005", 5, m_support=1, e_support=1),
        _case("0012", 12, m_support=1, e_support=1),
    ])
    initial["training_epoch"] = 1
    health.record_initial(initial)
    saved = health.state_dict()
    restored = MemberRoutingHealth(provider, {
        "enabled": True, "observation_epochs": 12, "later_every": 10,
        "panel_module_counts": [1, 2, 5, 12], "pause_on_collapse": True,
    })
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    assert set(saved) == {
        "schema", "panel_case_ids", "initial_observation", "last_observation",
        "last_observed_epoch", "previously_active_paths", "clean_stop",
    }

    reason = {"kind": "fixture_collapse", "path": "M"}
    written = request_clean_stop(tmp_path, epoch=2, reason=reason)
    payload = json.loads((tmp_path / "CLEAN_STOP_REQUEST.json").read_text())
    assert written["written"] is True
    assert payload["schema"] == "ordinary_clean_stop_request_v1"
    assert payload["health_evidence"] == reason
    original = (tmp_path / "CLEAN_STOP_REQUEST.json").read_bytes()
    preserved = request_clean_stop(tmp_path, epoch=3, reason={"kind": "newer"})
    assert preserved == {"written": False, "existing_request_preserved": True}
    assert (tmp_path / "CLEAN_STOP_REQUEST.json").read_bytes() == original
