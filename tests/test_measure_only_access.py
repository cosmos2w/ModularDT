"""The selected-row control keeps its scientific scope and checkpoint contract."""
from copy import deepcopy
from dataclasses import replace
from io import BytesIO

import pytest
import torch

from honf_forward_core.interaction_reused_member_routing import (
    AnchoredMemberRoutingCore, DetailAnchoredMemberRoutingCore,
    HyperedgeNonlinearReadCore, anchored_core_from_config,
)
from honf_forward_core.interaction_reused_member_routing.anchored import _measure_only_access_directory
from honf_runtime.contextual_frontier_evaluation import _contextual_checkpoint_parts
from tests.anchored_fixtures import _base_config
from tests.test_e_anchored_packed_execution import dense_oracle, fixture


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _model_and_scene(field_count=7):
    _, scene, query = fixture("local_e_cover")
    scene = replace(scene, module_measure=torch.tensor([[0.3, 1.4, 3.1], [2.2, 0., 0.]],
                                                       dtype=torch.double))
    base = {**_base_config(), "output_dim": field_count}
    model = DetailAnchoredMemberRoutingCore(base, query_tile_size=4, retained_mass=.73).double()
    return model, scene, query


def _assert_rows(learned, measured, scene):
    found_changed = False
    for mode in ("full", "restricted"):
        source, result = getattr(learned, mode), getattr(measured, mode)
        for name in ("offsets", "member_index", "original_id", "is_module", "applied_exchange", "structural_self"):
            torch.testing.assert_close(getattr(source, name), getattr(result, name), rtol=0, atol=0)
        for batch in range(source.batch_size):
            physical = torch.cat((scene.module_measure[batch], scene.environment_measure[batch]))
            for anchor in range(source.anchor_count):
                original, control = source.row(batch, anchor), result.row(batch, anchor)
                torch.testing.assert_close(control.measure, physical[original.member_index], rtol=0, atol=0)
                for role in (original.is_module, ~original.is_module):
                    if not role.any():
                        continue
                    before, actual = original.access_weight[role], control.access_weight[role]
                    weights = physical[original.member_index[role]]
                    expected = before.sum() * weights / weights.sum()
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
                    torch.testing.assert_close(actual.sum(), before.sum(), rtol=1e-12, atol=1e-14)
                    found_changed |= bool(((actual - before).abs() > 1e-8).any())
    assert found_changed


@pytest.mark.parametrize("field_count", [2, 7])
def test_trained_control_and_fixed_state_intervention_preserve_support_and_context(field_count):
    learned, scene, query = _model_and_scene(field_count)
    config = {**learned.get_config(), "within_role_access_prior": "measure_only"}
    control = anchored_core_from_config(config).double()
    control.load_state_dict(learned.state_dict(), strict=True)
    learned.eval(); control.eval()
    reference, changed = learned.prepare(scene), control.prepare(scene)

    for name in ("anchored_applied_EM", "anchored_applied_EE", "member_keys_M", "member_keys_E",
                 "member_values_M", "member_values_E"):
        torch.testing.assert_close(getattr(reference, name), getattr(changed, name), rtol=0, atol=0)
    for route in ("mm", "em", "me", "ee"):
        source = reference.final_applied_context_stages[route]
        result = changed.final_applied_context_stages[route]
        assert source["snapshot"].stage_identity == result["snapshot"].stage_identity
        torch.testing.assert_close(source["incidence"], result["incidence"], rtol=0, atol=0)
    _assert_rows(reference.e_access_directory, changed.e_access_directory, scene)
    expected_entries = sum(getattr(changed.e_access_directory, mode).member_index.numel()
                           for mode in ("full", "restricted"))
    assert changed.work_counts["prepared_e_directory_measure_only_entries"] == expected_entries

    baseline = learned.decode_paired(reference, query, return_trace=True)
    intervened = learned.decode_paired(reference, query, intervention={"within_role_access_prior": "measure_only"},
                                       return_trace=True)
    trained = control.decode_paired(changed, query, return_trace=True)
    assert intervened[2]["intervention_measure_only_access_entries"] == expected_entries
    for original, fixed, actual in zip(baseline[:2], intervened[:2], trained[:2]):
        assert actual.joint_fields.shape == (scene.batch_size, query.num_queries, field_count)
        torch.testing.assert_close(fixed.joint_fields, actual.joint_fields, rtol=0, atol=0)
        for name in ("anchor_read_indices", "anchor_read_member_index", "anchor_read_original_id", "anchor_read_is_module"):
            torch.testing.assert_close(original.trace[name], fixed.trace[name], rtol=0, atol=0)
        torch.testing.assert_close(fixed.trace["anchor_read_access_prior"], actual.trace["anchor_read_access_prior"], rtol=0, atol=0)
        for role in ("M", "E"):
            for metric in ("query_anchor_member_pairs", "query_anchor_coalesced_value_pairs"):
                assert original.work_counts[f"{metric}_{role}"] == fixed.work_counts[f"{metric}_{role}"]
            assert baseline[2][f"query_unique_union_raw_score_pairs_{role}"] == intervened[2][f"query_unique_union_raw_score_pairs_{role}"]


def test_measure_only_training_checkpoint_and_finite_gradients():
    model, scene, query = _model_and_scene()
    historical = model.get_config()
    assert "within_role_access_prior" not in historical
    explicit = DetailAnchoredMemberRoutingCore.from_config({**historical, "within_role_access_prior": "learned"}).double()
    assert explicit.get_config() == historical
    explicit.load_state_dict(model.state_dict(), strict=True)
    model.eval(); explicit.eval()
    torch.testing.assert_close(model.decode(model.prepare(scene), query).joint_fields,
                               explicit.decode(explicit.prepare(scene), query).joint_fields, rtol=0, atol=0)

    control = anchored_core_from_config({**historical, "within_role_access_prior": "measure_only"}).double().train()
    control.load_state_dict(model.state_dict(), strict=True)
    assert list(control.state_dict()) == list(model.state_dict())
    optimizer = torch.optim.AdamW(control.parameters(), lr=1e-3)
    full, restricted, _ = control.decode_paired(control.prepare(scene), query)
    loss = full.joint_fields.square().mean() + restricted.joint_fields.square().mean()
    loss.backward()
    grads = [param.grad for param in control.parameters() if param.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert control.member_router.joint_head.weight.grad.abs().sum() > 0
    optimizer.step()

    stream = BytesIO()
    torch.save({"config": control.get_config(), "model": control.state_dict(), "optimizer": optimizer.state_dict()}, stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=False)
    restored = anchored_core_from_config(checkpoint["config"]).double()
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    assert restored.get_config() == checkpoint["config"]
    assert len(restored_optimizer.state) == len(optimizer.state) > 0
    for current, rebuilt in zip(optimizer.state.values(), restored_optimizer.state.values()):
        assert current.keys() == rebuilt.keys()
        for name in current:
            if torch.is_tensor(current[name]):
                torch.testing.assert_close(current[name], rebuilt[name], rtol=0, atol=0)
            else:
                assert current[name] == rebuilt[name]
    for name, value in control.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    control.eval(); restored.eval()
    torch.testing.assert_close(control.decode(control.prepare(scene), query).joint_fields,
                               restored.decode(restored.prepare(scene), query).joint_fields, rtol=0, atol=0)


def test_control_read_matches_independent_dense_execution_and_gradients():
    _, scene, query = _model_and_scene()
    base = {**_base_config(), "output_dim": 7}
    model = AnchoredMemberRoutingCore(base, locator_mode="local_e_cover", query_tile_size=4,
                                      retained_mass=.73, within_role_access_prior="measure_only").double().eval()
    prepared = model.prepare(scene)
    parameters = tuple(model.parameters()) + (query.coordinates,)
    for mode in ("full", "restricted"):
        actual = model.decode(prepared, query, intervention={"read_mode": mode}).joint_fields
        expected, _ = dense_oracle(model, prepared, query, mode)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        actual_grad = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True, allow_unused=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), parameters, retain_graph=True, allow_unused=True)
        for left, right in zip(actual_grad, expected_grad):
            assert (left is None) == (right is None)
            if left is not None:
                torch.testing.assert_close(left, right, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("core", [AnchoredMemberRoutingCore, DetailAnchoredMemberRoutingCore, HyperedgeNonlinearReadCore])
def test_all_anchored_saved_core_schemas_reconstruct_control(core):
    kwargs = {"hyperedge_mode": "G"} if core is HyperedgeNonlinearReadCore else {}
    original = core(_base_config(), within_role_access_prior="measure_only", **kwargs)
    restored = anchored_core_from_config(original.get_config())
    assert restored.get_config() == original.get_config()
    restored.load_state_dict(original.state_dict(), strict=True)


def test_undefined_measures_and_incompatible_interventions_are_explicit():
    model, scene, query = _model_and_scene()
    prepared = model.eval().prepare(scene)
    with pytest.raises(ValueError, match="cannot change the selected members"):
        model.decode(prepared, query, intervention={"within_role_access_prior": "measure_only",
                                                    "access_intervention": "self_only"})
    with pytest.raises(ValueError, match="must be 'measure_only'"):
        model.decode(prepared, query, intervention={"within_role_access_prior": "unknown"})
    with pytest.raises(ValueError, match="zero physical measure"):
        _measure_only_access_directory(prepared.e_access_directory,
                                       replace(scene, module_measure=torch.zeros_like(scene.module_measure)), ("full",))
    with pytest.raises(ValueError, match="All-native"):
        DetailAnchoredMemberRoutingCore(_base_config(), reader_capability="all_native", within_role_access_prior="measure_only")


def _checkpoint_payload():
    model = DetailAnchoredMemberRoutingCore(_base_config(), input_dataset="thermal")
    core = model.get_config()
    arm = core["architecture"]
    provider = {"schema": "thermal_r14_detail_anchored_member_routing_provider_v1", "architecture": arm,
                "member_routing_schema": core["schema"],
                "adapter_state": {"schema": "thermal_interaction_reused_detail_anchored_member_routing_adapter_v1", "core": core},
                "recipe": {"member_routing": {"arm": arm, "model": deepcopy(core)}}}
    return {"arm": arm, "experiment_identity": {"workflow": "interaction_reused_detail_anchored_member_routing_training_v1",
                                                   "provider_identity": provider}}


def test_checkpoint_evaluator_compares_recorded_access_setting():
    payload = _checkpoint_payload()
    _contextual_checkpoint_parts(payload)
    provider = payload["experiment_identity"]["provider_identity"]
    provider["recipe"]["member_routing"]["model"]["within_role_access_prior"] = "measure_only"
    with pytest.raises(ValueError, match="identities must agree"):
        _contextual_checkpoint_parts(payload)
    provider["adapter_state"]["core"]["within_role_access_prior"] = "measure_only"
    _contextual_checkpoint_parts(payload)
    provider["adapter_state"]["core"]["within_role_access_prior"] = "unknown"
    provider["recipe"]["member_routing"]["model"]["within_role_access_prior"] = "unknown"
    with pytest.raises(ValueError, match="identities must agree"):
        _contextual_checkpoint_parts(payload)
