"""Record and replay the actual numerical reader, including projected actions."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from honf_forward_core.evaluation.reference_access import FixedReferenceAccessReplay
from honf_forward_core.evaluation.typed_work_evidence import TypedWorkEvidenceRecorder, compare_native_access
from honf_forward_core.interface_fields.adaptive_interaction_cover import InteractionContext
from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
from honf_forward_core.interface_fields.types import EncodedInterfaceCase


def _fixture():
    torch.manual_seed(43)
    encoded = EncodedInterfaceCase(torch.randn(1, 3, 8), torch.randn(1, 5, 8), torch.randn(1, 8),
        torch.rand(1, 3, 2), torch.rand(1, 5, 2), torch.ones(1, 3), torch.randn(1, 3, 3),
        None, torch.ones(1, 5), torch.ones(2))
    backend = TypedHypergraphField(8, 12, 2, 2, architecture="faithful_receiver_hypergraph_honf",
        spatial_dim=2, module_characteristic_length=.03).eval()
    queries, features = torch.rand(1, 7, 2), torch.rand(1, 7, 6)
    with torch.no_grad():
        backend.read(backend.prepare(encoded, encoded.module_tokens), encoded, queries, features)
        for gain in backend.control_gain.values():
            gain.weight.normal_(0, .3)
            gain.bias.normal_(0, .1)
        backend.control_score.weight.normal_(0, .3)
        backend.control_score.bias.normal_(0, .1)
    return backend, encoded, queries, features


def _run(backend, encoded, queries, features):
    values = []
    for phase in range(3):
        state = backend.prepare(encoded, encoded.module_tokens, interaction_context=InteractionContext(phase=f"P{phase}"))
        pieces = []
        for start in range(0, 7, 3):
            pieces.append(backend.read(state, encoded, queries[:, start:start+3], features[:, start:start+3])[0])
        values.append(torch.cat(pieces, 1))
    return values


def test_actual_projected_access_calls_are_recorded_once_without_full_control_materialization():
    backend, encoded, queries, features = _fixture()
    with torch.no_grad():
        expected = _run(backend, encoded, queries, features)
        with TypedWorkEvidenceRecorder(backend) as recorded:
            actual = _run(backend, encoded, queries, features)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    prefixes = [key.removesuffix("/support") for key in recorded.arrays if key.endswith("/support")]
    assert len(prefixes) == 27  # three preparations + three QM/QE chunks per phase
    for prefix in prefixes:
        assert f"{prefix}/control_probe" not in recorded.arrays
        assert f"{prefix}/projected_action_probe" in recorded.arrays
        channels = 4 if "/QE/" in prefix else 1
        assert recorded.arrays[f"{prefix}/projected_action_probe"].shape[-1] == channels
        assert any(key.startswith(f"{prefix}/executor/") for key in recorded.arrays)
    assert all(name not in backend.__dict__ for name in ("_access", "_numerical_access", "prepare", "_ledger"))


def test_full_control_diagnostic_recorder_preserves_legacy_keys_and_matches_small_actions():
    backend, encoded, queries, features = _fixture()
    with torch.no_grad():
        with TypedWorkEvidenceRecorder(backend) as projected:
            expected = _run(backend, encoded, queries, features)
        backend.control_execution = "full_control"
        with TypedWorkEvidenceRecorder(backend) as diagnostic:
            actual = _run(backend, encoded, queries, features)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)
    comparison = compare_native_access(projected.arrays, diagnostic.arrays)
    assert len(comparison) == 15
    assert all(row["projected_action_probe_equal"] and row["projected_action_receiver_summary_equal"]
               for row in comparison.values())
    assert all(row["control_probe_equal"] is None for row in comparison.values())
    for key, value in diagnostic.arrays.items():
        if key.endswith("/control_probe"):
            assert value.shape[-1] == 16


@pytest.mark.parametrize("mode", ["control_identity", "normal_fixed_actions", "full_access_fixed_controls", "geometry_reference_actions"])
def test_projected_reference_replay_is_effective_and_restores_numerical_reader(mode):
    backend, encoded, queries, features = _fixture()
    with torch.no_grad():
        with TypedWorkEvidenceRecorder(backend) as normal:
            original = _run(backend, encoded, queries, features)
        with FixedReferenceAccessReplay(backend, normal.arrays, mode=mode) as replay, TypedWorkEvidenceRecorder(backend) as result:
            assert backend.control_execution == "full_control"
            intervened = _run(backend, encoded, queries, features)
    assert replay.seen == replay.expected and len(replay.seen) == 27
    assert backend.control_execution == "projected" and backend.plan_intervention == "normal"
    assert all(name not in backend.__dict__ for name in ("_access", "_numerical_access", "prepare", "_ledger"))
    if mode == "control_identity":
        assert any((a-b).abs().max() > 1e-5 for a, b in zip(original, intervened))
        assert all(np.count_nonzero(value) == 0 for key, value in result.arrays.items()
                   if key.endswith("/projected_action_probe"))
    elif mode in {"normal_fixed_actions", "full_access_fixed_controls"}:
        comparison = compare_native_access(normal.arrays, result.arrays)
        assert all(row["projected_action_probe_equal"] for row in comparison.values())
        if mode == "normal_fixed_actions":
            assert all(row["max_absolute_weight_change"] == 0 and row["changed_pairs"] == 0
                       for row in comparison.values())
            for a, b in zip(original, intervened):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_normal_actions_stay_fixed_when_live_physical_source_states_change():
    backend, encoded, queries, features = _fixture()
    with torch.no_grad():
        with TypedWorkEvidenceRecorder(backend) as normal:
            original = _run(backend, encoded, queries, features)
        changed = replace(encoded, module_tokens=encoded.module_tokens + 0.4, env_tokens=encoded.env_tokens - 0.3)
        with (
            FixedReferenceAccessReplay(backend, normal.arrays, mode="normal_fixed_actions"),
            TypedWorkEvidenceRecorder(backend) as result,
        ):
            actual = _run(backend, changed, queries, features)
    assert any((a-b).abs().max() > 1e-5 for a, b in zip(original, actual))
    comparison = compare_native_access(normal.arrays, result.arrays)
    assert all(row["max_absolute_weight_change"] == 0 and row["changed_pairs"] == 0 and row["projected_action_probe_equal"]
               for row in comparison.values())
    assert backend.control_execution == "projected" and backend.plan_intervention == "normal"


def test_normal_action_replay_rejects_projected_control_drift_and_restores():
    backend, encoded, queries, features = _fixture()
    with torch.no_grad():
        with TypedWorkEvidenceRecorder(backend) as normal:
            _run(backend, encoded, queries, features)
        normal.arrays["access/P0/MM/00000/projected_action_probe"].flat[0] += 1
        with (
            pytest.raises(ValueError, match="reconstructed control projected_action_probe mismatch"),
            FixedReferenceAccessReplay(backend, normal.arrays, mode="normal_fixed_actions"),
        ):
            _run(backend, encoded, queries, features)
    assert backend.control_execution == "projected" and backend.plan_intervention == "normal"
    assert all(name not in backend.__dict__ for name in ("_access", "_numerical_access", "prepare", "_ledger"))


def test_explicit_diagnostic_plan_rebinding_rebuilds_prepared_actions_once():
    backend, encoded, queries, features = _fixture()
    with torch.no_grad():
        state = backend.prepare(encoded, encoded.module_tokens)
        plan = state["hypergraph_plan"]
        membership = plan.memberships["QE"].clone()
        membership[..., 1] = 0
        state["hypergraph_plan"] = replace(plan, memberships={**plan.memberships, "QE": membership})
        calls = []
        original = backend._prepare_actions

        def rebuild(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        backend._prepare_actions = rebuild
        result, _ = backend.read(state, encoded, queries, features)
        repeated, _ = backend.read(state, encoded, queries, features)
        assert len(calls) == 1 and state["hypergraph_action_plan"] is state["hypergraph_plan"]
        reference, _ = backend.read({**state, "hypergraph_actions": None}, encoded, queries, features)
    torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(repeated, result, rtol=0, atol=0)
