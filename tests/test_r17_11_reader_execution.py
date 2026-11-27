"""Work and checkpoint-replay parity for the original-member reader."""
from __future__ import annotations

import copy
from dataclasses import fields, replace
import os
import types

import pytest
import torch

from honf_forward_core.interaction_reused_member_routing import DetailAnchoredMemberRoutingCore
from honf_forward_core.interaction_reused_member_routing.anchored import anchored_checkpoint_integer_plans
from honf_forward_core.content_hypergraph.query_chunks_v1 import select_query_rows_v1
from tests.anchored_fixtures import _base_config, _query, _scene


DEVICE = torch.device(os.environ.get("HONF_READER_TEST_DEVICE", "cpu"))


@pytest.fixture(scope="module", autouse=True)
def _deterministic_execution():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _fixture_inputs(high_member=False):
    scene = _scene()
    if high_member:
        module_fields = ("module_features", "module_coordinates", "module_mask", "module_measure",
                         "module_ids", "local_radius", "module_characteristic_lengths")
        environment_fields = ("environment_features", "environment_coordinates", "environment_mask",
                              "environment_measure", "environment_ids", "environment_characteristic_lengths")
        updates = {name: getattr(scene, name).repeat_interleave(8, dim=1)
                   for name in module_fields + environment_fields}
        for prefix in ("module", "environment"):
            mask = updates[f"{prefix}_mask"]
            ids = torch.arange(mask.shape[1]).expand_as(mask) + (100 if prefix == "module" else 1000)
            updates[f"{prefix}_ids"] = ids.masked_fill(~mask, -1)
        scene = replace(scene, **updates)
    scene = replace(scene, **{f.name: value.to(DEVICE) for f in fields(scene)
                              if torch.is_tensor(value := getattr(scene, f.name))})
    query = _query()
    coordinates = query.coordinates.to(DEVICE).detach().requires_grad_(True)
    mask = query.query_mask.to(DEVICE).clone()
    mask[1, -1] = False
    shape = coordinates.shape[:2]
    anchors = torch.stack((torch.zeros(shape, dtype=torch.long, device=DEVICE),
                           torch.ones(shape, dtype=torch.long, device=DEVICE)), -1)
    weights = torch.stack((coordinates[..., 0].sigmoid(), 1 - coordinates[..., 0].sigmoid()), -1).detach()
    # Preserve exact-zero support and exercise subnormal/underflow behavior.
    weights[:, 0] = weights.new_tensor([1., 0.])
    weights[:, 1] = weights.new_tensor([torch.finfo(weights.dtype).tiny / 2, 1.])
    weights[~mask] = 0
    anchors[~mask] = -1
    weights.requires_grad_(True)
    query = replace(query, coordinates=coordinates, query_mask=mask, anchor_indices=anchors,
                    anchor_weights=weights, query_ids=query.query_ids.to(DEVICE))
    return scene, query


@pytest.fixture(scope="module")
def trained_checkpoint():
    path = os.environ.get("HONF_READER_TRAINED_CHECKPOINT")
    return None if not path else torch.load(path, map_location="cpu", weights_only=False)


def _model_and_optimizer(state, checkpoint):
    if state == "trained":
        if checkpoint is None:
            pytest.skip("Set HONF_READER_TRAINED_CHECKPOINT to exercise a saved trained state.")
        config = checkpoint["experiment_identity"]["recipe"]["member_routing"]["model"]
        model = DetailAnchoredMemberRoutingCore.from_config(config)
        model.load_state_dict({key.removeprefix("core."): value for key, value in
                               checkpoint["model_state_dict"].items() if key.startswith("core.")})
    else:
        model = DetailAnchoredMemberRoutingCore({**_base_config(), "output_dim": 7}, retained_mass=.5)
    model = model.to(DEVICE).train()
    model.query_tile_size = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    if state == "trained":
        saved = checkpoint["optimizer_state_by_name"]["state_by_name"]
        for name, parameter in model.named_parameters():
            if "core." + name in saved:
                optimizer.state[parameter] = {
                    key: (value.detach().clone().to("cpu" if key == "step" else DEVICE)
                          if torch.is_tensor(value) else copy.deepcopy(value))
                    for key, value in saved["core." + name].items()
                }
    return model, optimizer


def _one_update(state, checkpoint, mode, high_member, use_plan):
    model, optimizer = _model_and_optimizer(state, checkpoint)
    scene, query = _fixture_inputs(high_member)
    with anchored_checkpoint_integer_plans(use_plan):
        prepared = model.prepare(scene)
        if mode == "paired":
            first, second, _ = model.decode_paired(prepared, query)
            outputs = (first.joint_fields, second.joint_fields)
        else:
            with model.read_execution(mode):
                outputs = (model.decode(prepared, query).joint_fields,)
    # Keep the component order explicit and exercise every configured field.
    terms = [((output[..., index] - .07 * (index + 1)).square().mean(), view, index)
             for view, output in enumerate(outputs) for index in range(output.shape[-1])]
    total = sum((1 + index * .1 + view * .03) * value for value, view, index in terms)
    total.backward()
    gradients = {name: None if value.grad is None else value.grad.detach().clone()
                 for name, value in model.named_parameters()}
    input_gradients = (query.coordinates.grad.detach().clone(), query.anchor_weights.grad.detach().clone())
    optimizer.step()
    return ([value.detach().clone() for value in outputs],
            [value.detach().clone() for value, _, _ in terms], gradients, input_gradients,
            {name: value.detach().clone() for name, value in model.named_parameters()},
            {name: {key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in optimizer.state[parameter].items()}
             for name, parameter in model.named_parameters() if parameter in optimizer.state})


def _compare(actual, expected):
    maxima = {"outputs": 0., "terms": 0., "gradients": 0., "inputs": 0., "parameters": 0., "optimizer": 0.}

    def close(left, right, section, name):
        assert (left is None) == (right is None), name
        if left is None:
            return
        if not torch.is_tensor(left):
            assert left == right, name
            return
        if left.numel():
            assert torch.isfinite(left).all() and torch.isfinite(right).all(), name
            maxima[section] = max(maxima[section], (left - right).abs().max().item())
        torch.testing.assert_close(left, right, rtol=3e-5, atol=2e-6, msg=name)

    for section, a, b in zip(("outputs", "terms", "gradients", "inputs", "parameters", "optimizer"),
                             actual, expected):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for name in a:
                if isinstance(a[name], dict):
                    assert a[name].keys() == b[name].keys()
                    for key in a[name]:
                        close(a[name][key], b[name][key], section, name + "/" + key)
                else:
                    close(a[name], b[name], section, name)
        else:
            for index, (left, right) in enumerate(zip(a, b)):
                close(left, right, section, str(index))
    return maxima


@pytest.mark.parametrize("state,high_member,mode", [
    ("fresh", False, "full"), ("fresh", False, "restricted"), ("fresh", False, "paired"),
    ("fresh", True, "paired"), ("trained", True, "full"),
    ("trained", True, "restricted"), ("trained", True, "paired"),
])
def test_checkpoint_plan_matches_self_repeat_and_optimizer(state, high_member, mode, trained_checkpoint):
    reference = _one_update(state, trained_checkpoint, mode, high_member, False)
    repeat = _one_update(state, trained_checkpoint, mode, high_member, False)
    candidate = _one_update(state, trained_checkpoint, mode, high_member, True)
    repeat_difference = _compare(repeat, reference)
    candidate_difference = _compare(candidate, reference)
    print(f"reader parity device={DEVICE} state={state} high_member={high_member} mode={mode} "
          f"self_repeat={repeat_difference} candidate={candidate_difference}")


def test_single_reader_executes_only_requested_work_and_api():
    model, _ = _model_and_optimizer("fresh", None)
    model.eval()
    scene, query = _fixture_inputs(high_member=True)
    model.query_tile_size = query.num_queries
    prepared = model.prepare(scene)
    heads = []
    hook = model.member_router.joint_head.register_forward_hook(lambda *_: heads.append(1))
    try:
        singles = {}
        for mode in ("full", "restricted"):
            heads.clear()
            with model.read_execution(mode):
                singles[mode] = model.decode(prepared, query)
            assert len(heads) == 1
            counts = singles[mode].work_counts
            assert counts[f"query_{mode}_reader_head_rows"] == query.batch_size * query.num_queries
            other = "restricted" if mode == "full" else "full"
            assert counts[f"query_{other}_reader_head_rows"] == 0
            explicit = model.decode(prepared, query, intervention={"read_mode": mode})
            torch.testing.assert_close(singles[mode].joint_fields, explicit.joint_fields, rtol=0, atol=0)
        heads.clear()
        full, restricted, shared = model.decode_paired(prepared, query)
        assert len(heads) == 2
        for mode, output in (("full", full), ("restricted", restricted)):
            torch.testing.assert_close(singles[mode].joint_fields, output.joint_fields, rtol=3e-5, atol=2e-6)
            for role in ("M", "E"):
                assert (singles[mode].work_counts[f"query_anchor_value_reduction_pairs_{role}"] ==
                        output.work_counts[f"query_anchor_value_reduction_pairs_{role}"])
                assert (singles[mode].work_counts[f"query_unique_union_raw_score_pairs_{role}"] <=
                        shared[f"query_unique_union_raw_score_pairs_{role}"])
        assert (singles["restricted"].work_counts["query_anchor_member_pairs_E"] <
                singles["full"].work_counts["query_anchor_member_pairs_E"])
    finally:
        hook.remove()


def test_plan_contains_only_integers_and_replay_reuses_it():
    model, _ = _model_and_optimizer("fresh", None)
    scene, query = _fixture_inputs()
    model.query_tile_size = query.num_queries
    calls = []
    original = model._decode_tile

    def inspect(_self, *args, **kwargs):
        calls.append(kwargs.get("integer_plan"))
        return original(*args, **kwargs)

    model._decode_tile = types.MethodType(inspect, model)
    with anchored_checkpoint_integer_plans(True):
        full, restricted, _ = model.decode_paired(model.prepare(scene), query)
    (full.joint_fields.square().mean() + restricted.joint_fields.square().mean()).backward()
    assert len(calls) >= 2 and calls[0] is calls[1] and calls[0] is not None
    plan = calls[0]
    assert not plan["union_keys"].is_floating_point()
    for row in plan["memberships"].values():
        assert all(not value.is_floating_point() and not value.requires_grad for value in row.values())
    assert query.anchor_weights.grad is not None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters() if "applied_incidence" in name)


@pytest.mark.parametrize("state,mode", [("fresh", "full"), ("fresh", "restricted"),
                                        ("trained", "full"), ("trained", "restricted")])
def test_selected_prediction_and_gradients_are_independent_of_companions(state, mode, trained_checkpoint):
    model, _ = _model_and_optimizer(state, trained_checkpoint)
    model.query_tile_size = 3
    scene, original_query = _fixture_inputs()

    def selected_case(scenario):
        coordinates = original_query.coordinates.detach().clone()
        weights = original_query.anchor_weights.detach().clone()
        if scenario == "changed_queries":
            coordinates[:, [0, 1, 3]] += 1.25
        coordinates.requires_grad_(True)
        weights.requires_grad_(True)
        query = replace(original_query, coordinates=coordinates, anchor_weights=weights)
        current_scene, target_index = scene, 2
        if scenario == "reordered":
            query = select_query_rows_v1(query, torch.tensor([2, 3, 0, 1], device=DEVICE))
            target_index = 0
        elif scenario == "changed_scene":
            module = scene.module_features.clone()
            environment = scene.environment_features.clone()
            operating = scene.operating_features.clone()
            module[1] += 3
            environment[1] -= 2
            operating[1] += 1
            current_scene = replace(scene, module_features=module, environment_features=environment,
                                    operating_features=operating)
        elif scenario == "alone":
            current_scene = replace(scene, **{f.name: value[:1] for f in fields(scene)
                                              if torch.is_tensor(value := getattr(scene, f.name))})
            query = replace(query, **{name: getattr(query, name)[:1] for name in
                                      ("coordinates", "query_mask", "query_ids", "anchor_indices", "anchor_weights")})
        with model.read_execution(mode):
            output = model.decode(model.prepare(current_scene), query).joint_fields[0, target_index]
        scalar = (output * torch.arange(1, output.numel() + 1, device=DEVICE)).sum()
        named = tuple(model.named_parameters())
        gradients = torch.autograd.grad(scalar, [parameter for _, parameter in named] + [coordinates, weights],
                                        allow_unused=True)
        return output.detach(), {name: gradient for (name, _), gradient in zip(named, gradients)}, (
            gradients[-2][0, 2].detach(), gradients[-1][0, 2].detach())

    reference = selected_case("reference")
    for scenario in ("reordered", "changed_queries", "changed_scene", "alone"):
        actual = selected_case(scenario)
        torch.testing.assert_close(actual[0], reference[0], rtol=5e-5, atol=4e-6, msg=scenario)
        for name in reference[1]:
            left, right = actual[1][name], reference[1][name]
            assert (left is None) == (right is None), (scenario, name)
            if left is not None:
                torch.testing.assert_close(left, right, rtol=5e-5, atol=4e-6, msg=f"{scenario}: {name}")
        for left, right in zip(actual[2], reference[2]):
            torch.testing.assert_close(left, right, rtol=5e-5, atol=4e-6, msg=scenario)


def test_checkpoint_plan_scope_restores_after_exception():
    from honf_forward_core.interaction_reused_member_routing.anchored import _checkpoint_integer_plans
    previous = _checkpoint_integer_plans.get()
    with pytest.raises(RuntimeError), anchored_checkpoint_integer_plans(not previous):
        assert _checkpoint_integer_plans.get() is not previous
        raise RuntimeError("stop")
    assert _checkpoint_integer_plans.get() is previous
    with pytest.raises(ValueError), anchored_checkpoint_integer_plans("false"):
        pass
