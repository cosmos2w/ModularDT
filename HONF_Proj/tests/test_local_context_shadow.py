"""Local organizer derivatives reuse physical values and restore eligible QE donors."""

import os
from copy import deepcopy

import pytest
import torch
from torch import nn

from honf_forward_core.training.hypergraph_shadow import (
    bridge_local_context,
    hard_value_soft_hypergraph_forward,
    local_attention_context,
    local_modulated_sum,
)


def test_completed_attention_restores_omitted_donor_without_physical_gradients():
    query = torch.zeros(1, 1, 1, 2, requires_grad=True)
    key = torch.zeros(1, 1, 2, 2, requires_grad=True)
    value = torch.tensor([[[[1., 0.], [0., 3.]]]], requires_grad=True)
    bias = torch.zeros(1, 1, 1, 2, requires_grad=True)
    hard_support = torch.tensor([[[True, False]]])
    hard_weight = torch.tensor([[[1., 0.]]])
    soft_logit = torch.tensor(-2., dtype=torch.float64, requires_grad=True)
    soft_weight = torch.stack((soft_logit.new_tensor(1.), soft_logit.exp())).reshape(1, 1, 2)
    soft_support = torch.ones_like(hard_support)
    score = torch.zeros(1, 1, 2, 1)
    gain = torch.zeros_like(score)
    measure = torch.ones(1, 2)
    hard = local_attention_context(query, key, value, bias, hard_weight, hard_support, score, gain, measure)
    soft = local_attention_context(query.detach(), key.detach(), value.detach(), bias.detach(),
                                   soft_weight, soft_support, score, gain, measure)
    bridged = bridge_local_context(hard, soft)
    torch.testing.assert_close(bridged, hard, rtol=0, atol=0)
    # The omitted donor contributes to component one only in the soft context.
    organizer_gradient = torch.autograd.grad(bridged[..., 1].sum(), soft_logit, retain_graph=True)[0]
    assert organizer_gradient > 0
    physical = (query, key, value, bias)
    actual = torch.autograd.grad(bridged.square().sum(), physical, retain_graph=True)
    reference = torch.autograd.grad(hard.square().sum(), physical)
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_fp64_permission_reaches_log_before_physical_cast():
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 2, 2)
    value = torch.tensor([[[[1., 0.], [0., 3.]]]])
    bias = torch.zeros(1, 1, 1, 2)
    weight = torch.tensor([[[1., 1e-100]]], dtype=torch.float64, requires_grad=True)
    control = torch.zeros(1, 1, 2, 1)
    context = local_attention_context(query, key, value, bias, weight, torch.ones(1, 1, 2, dtype=torch.bool),
                                      control, control, torch.ones(1, 2))
    context.sum().backward()
    assert torch.isfinite(context).all()
    assert torch.isfinite(weight.grad).all()


def test_weighted_message_bridge_keeps_physical_and_projection_gradients():
    generator = torch.Generator().manual_seed(143)
    messages = torch.randn(2, 3, 4, 5, generator=generator, requires_grad=True)
    projection = nn.Linear(3, 1)
    control_hard = torch.randn(2, 3, 4, 3, generator=generator)
    control_soft = control_hard.clone().double().requires_grad_()
    hard_gain = projection(control_hard)
    soft_gain = torch.nn.functional.linear(control_soft, projection.weight.detach().double(),
                                            projection.bias.detach().double())
    hard_weight = torch.rand(2, 3, 4, generator=generator)
    soft_weight = hard_weight.double().requires_grad_()
    measure = torch.rand(2, 1, 4, generator=generator)
    hard = local_modulated_sum(messages, hard_weight, hard_gain, measure)
    soft = local_modulated_sum(messages.detach(), soft_weight, soft_gain, measure.detach())
    output = nn.Linear(5, 2)
    expected = output(hard)
    actual = output(bridge_local_context(hard, soft))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    physical = (messages, projection.weight, projection.bias, output.weight, output.bias)
    actual_grad = torch.autograd.grad(actual.square().sum(), physical, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), physical, retain_graph=True)
    for got, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, reference, rtol=0, atol=0)
    organizer_grad = torch.autograd.grad(actual.square().sum(), (control_soft, soft_weight))
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in organizer_grad)


def test_unknown_gradient_policy_rejected():
    with pytest.raises(ValueError, match="Unknown organizer gradient policy"):
        hard_value_soft_hypergraph_forward(nn.Linear(1, 1), gradient_policy="silently_changed")


class _OnePassWrapper(nn.Module):
    def __init__(self, *, checkpointing=False):
        super().__init__()
        from honf_forward_core.interface_fields.typed_hypergraph_field import TypedHypergraphField
        self.core = nn.Module()
        self.core.backend = TypedHypergraphField(8, 12, 2, 2,
            architecture="faithful_receiver_hypergraph_honf", spatial_dim=2,
            module_characteristic_length=.1, activation_checkpointing=checkpointing,
            options={"organizer_dim": 16})
        self.physical = nn.Linear(8, 1)
        self.calls = 0

    def forward(self, encoded, query, features):
        self.calls += 1
        prepared = self.core.backend.prepare(encoded, encoded.module_tokens)
        context, auxiliary = self.core.backend.read(prepared, encoded, query, features)
        return {"pred_field": self.physical(context), "interaction_aux": auxiliary,
                "_prepared": prepared}


def _case():
    from honf_forward_core.interface_fields.types import EncodedInterfaceCase
    generator = torch.Generator().manual_seed(432)
    return EncodedInterfaceCase(
        module_tokens=torch.randn(2, 4, 8, generator=generator),
        env_tokens=torch.randn(2, 7, 8, generator=generator),
        global_token=torch.randn(2, 8, generator=generator),
        module_centers=torch.rand(2, 4, 2, generator=generator),
        env_coords=torch.rand(2, 7, 2, generator=generator),
        module_present=torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]]),
        module_features=torch.randn(2, 4, 3, generator=generator), env_features=None,
        env_weights=torch.tensor([[1., .5, 2., 1., 3., .75, 1.]]).expand(2, -1),
        coordinate_scale=torch.ones(1, 1, 2))


@pytest.mark.parametrize("checkpointing", [False, True])
def test_single_wrapper_physical_input_gradients_and_all_route_organizer_scopes(checkpointing):
    from collections import Counter
    from dataclasses import replace

    encoded = _case()
    model = _OnePassWrapper(checkpointing=checkpointing).train()
    model.core.backend.organizer.set_epoch(301)
    model.core.backend.organizer.eval()
    query, features = torch.rand(2, 5, 2), torch.rand(2, 5, 6)
    model(encoded, query, features)
    with torch.no_grad():
        for gain in model.core.backend.control_gain.values():
            gain.weight.normal_(0, .1)
        model.core.backend.control_score.weight.normal_(0, .1)
        for strength in model.core.backend.organizer.geometry_strength.values():
            strength.fill_(2.)
        for head in model.core.backend.organizer.control_heads.values():
            head[-1].weight.normal_(0, .1)
    reference = deepcopy(model)
    physical_inputs = []
    predictions = []
    fine_counts = Counter()
    handles = []
    for name in ("mm_message", "me_message", "em_message", "query_module_message", "env_geometry_bias"):
        handles.append(getattr(model.core.backend, name).register_forward_hook(
            lambda module, args, output, name=name: fine_counts.update((name,))))
    for index, current in enumerate((reference, model)):
        states = encoded.module_tokens.clone().requires_grad_()
        env = encoded.env_tokens.clone().requires_grad_()
        coords = query.clone().requires_grad_()
        record = replace(encoded, module_tokens=states, env_tokens=env)
        before = current.calls
        output = (current(record, coords, features) if index == 0 else
                  hard_value_soft_hypergraph_forward(current, record, coords, features,
                      gradient_policy="local_context_shadow_v1"))
        assert current.calls == before + 1
        if index == 1:
            assert output["campaign_shadow_calls"] == 0
            assert all(count == 1 for count in fine_counts.values()) and len(fine_counts) == 5
            assert output["_prepared"]["hypergraph_training_gradient_mode"] == "local_context_shadow_v1"
            assert current.core.backend.training_gradient_mode == "whole_wrapper_shadow_v1"
            assert output["campaign_structural_cost"].requires_grad
            current.core.backend.training_gradient_mode = "disabled_for_replay_test"
            current.core.backend.permission_mode = "soft"
            for handle in handles:
                handle.remove()
        predictions.append(output["pred_field"].detach())
        output["pred_field"].square().sum().backward()
        physical_inputs.append((states.grad, env.grad, coords.grad))
    torch.testing.assert_close(predictions[1], predictions[0], rtol=0, atol=0)
    for got, expected in zip(physical_inputs[1], physical_inputs[0]):
        torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)
    parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        if ".organizer." not in name:
            torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=2e-5, atol=2e-6)
    organizer = model.core.backend.organizer
    for tau in ("MM", "ME", "EM", "QM", "QE"):
        gradients = [p.grad for p in organizer.control_heads[tau].parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    assert organizer.split_head.weight.grad is not None
    assert torch.isfinite(organizer.split_head.weight.grad).all()
    assert organizer.split_head.weight.grad.abs().sum() > 0


def test_local_soft_preparation_preserves_hard_rng_and_exploration_counter():
    encoded = _case()
    model = _OnePassWrapper().train()
    query, features = torch.rand(2, 5, 2), torch.rand(2, 5, 6)
    model(encoded, query, features)
    model.core.backend.organizer.set_epoch(201)
    reference = deepcopy(model)
    state = torch.random.get_rng_state()
    hard = reference(encoded, query, features)
    expected_rng = torch.random.get_rng_state()
    expected_counter = reference.core.backend.organizer.exercise_counter.clone()
    torch.random.set_rng_state(state)
    local = hard_value_soft_hypergraph_forward(model, encoded, query, features,
                                              gradient_policy="local_context_shadow_v1")
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert torch.equal(model.core.backend.organizer.exercise_counter, expected_counter)
    torch.testing.assert_close(local["pred_field"], hard["pred_field"], rtol=0, atol=0)


def test_entirely_omitted_hard_qe_row_keeps_soft_restoration_after_output_projection():
    import types
    from dataclasses import replace

    encoded = _case()
    model = _OnePassWrapper().train()
    model.core.backend.organizer.eval()
    query, features = torch.rand(2, 5, 2), torch.rand(2, 5, 6)
    model(encoded, query, features)
    with torch.no_grad():
        model.core.backend.control_gain["QE"].weight.normal_(0, .2)
        model.core.backend.control_score.weight.normal_(0, .2)
    original = model.core.backend._numerical_access

    def omit_qe(self, plan, receivers, mechanism, *args, **kwargs):
        result = original(plan, receivers, mechanism, *args, **kwargs)
        if mechanism == "QE" and not plan.strategy_data["soft"]:
            result = replace(result, weight=torch.zeros_like(result.weight), support=torch.zeros_like(result.support))
        return result

    model.core.backend._numerical_access = types.MethodType(omit_qe, model.core.backend)
    hard = model(encoded, query, features)["pred_field"]
    local = hard_value_soft_hypergraph_forward(model, encoded, query, features,
                                              gradient_policy="local_context_shadow_v1")["pred_field"]
    torch.testing.assert_close(local, hard, rtol=0, atol=0)
    local.square().sum().backward()
    gradients = [p.grad for p in model.core.backend.organizer.control_heads["QE"].parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


@pytest.mark.skipif(not os.environ.get("HONF_TREE_SHADOW_CHECKPOINT"),
                    reason="Needs an explicitly selected fixed-quarter Tree checkpoint")
def test_retained_tree_native_loss_physical_gradients_frozen_stage_a_and_work():
    """Bounded native test; caller explicitly selects device and native Q."""
    from pathlib import Path

    from channelthermal.data.collation import ChannelThermalBatchCollator
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.data.development_split import development_case_ids
    from channelthermal.evaluation.loading import load_model
    from channelthermal.training.campaign_work import CampaignForwardWork
    from channelthermal.training.epoch import assemble_channelthermal_loss_terms, make_model_inputs
    from torch.utils.data import DataLoader

    from honf_runtime.compat import recursive_to_device

    device = torch.device(os.environ.get("HONF_TREE_SHADOW_DEVICE", "cpu"))
    model, checkpoint = load_model(Path(os.environ["HONF_TREE_SHADOW_CHECKPOINT"]), device)
    config = checkpoint["train_config"]
    settings = config["dataset"]
    selected = development_case_ids(settings["development_subset"], "train")
    assert len(selected) == 150
    case_ids = os.environ.get("HONF_TREE_SHADOW_CASE_IDS", ",".join(selected[:2])).split(",")
    assert 1 <= len(case_ids) <= 8 and set(case_ids) <= set(selected)
    queries = int(os.environ.get("HONF_TREE_SHADOW_QUERIES", "17"))
    dataset = GlobalChannelThermalDataset(settings["packed_h5_path"], split="train", points_per_case=queries,
        normalize_inputs=settings["normalize_inputs"], normalize_targets=settings["normalize_targets"],
        random_point_sampling=False, seed=0, include_grid=False,
        normalizer=H5Normalizer(checkpoint["global_normalization_stats"]), case_ids=case_ids)
    batch = recursive_to_device(next(iter(DataLoader(dataset, batch_size=len(case_ids),
        collate_fn=ChannelThermalBatchCollator()))), device)
    inputs = make_model_inputs(batch, local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
                               return_predicted_port_outputs=True, return_port_global_consistency=True)
    model.train()
    model.set_training_progress(epoch=201, total_epochs=1000)
    model.core.backend.organizer.eval()  # Same hard topology without exploration noise.
    model.core.backend.activation_checkpointing = True
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    local_buffers = {name: value.detach().clone() for name, value in model.local_coupling.named_buffers()}
    physical = {name: p for name, p in model.named_parameters()
                if p.requires_grad and not name.startswith("core.backend.organizer.")}

    def criterion(output):
        return assemble_channelthermal_loss_terms(output, batch, model, config["loss"],
            local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
            effective_internal_temperature_weight=1., effective_interface_weight=.2,
            predicted_consistency_weight=.05)["loss_physical"]

    hard = model(**inputs)
    hard_loss = criterion(hard)
    expected_value = float(hard_loss.detach())
    expected_fields = {key: hard[key].detach().clone() for key in
                       ("pred_field", "pred_interface", "pred_internal_temperature")}
    hard_loss.backward()
    expected_grad = {name: None if p.grad is None else p.grad.detach().clone() for name, p in physical.items()}
    del hard, hard_loss
    model.zero_grad(set_to_none=True)
    with CampaignForwardWork(model.core) as work:
        local = hard_value_soft_hypergraph_forward(model, gradient_policy="local_context_shadow_v1", **inputs)
    snapshot = deepcopy(work.records)
    assert set(snapshot) == {"hard"} and set(snapshot["hard"]) == {"P0", "P1", "P2"}
    assert all(v["prepare_calls"] == 1 for v in snapshot["hard"].values())
    local_loss = criterion(local)
    assert float(local_loss.detach()) == pytest.approx(expected_value, rel=2e-6, abs=2e-7)
    for key, expected in expected_fields.items():
        torch.testing.assert_close(local[key], expected, rtol=0, atol=0)
    local_loss.backward()
    assert work.records == snapshot
    numerator = denominator = 0.
    for name, p in physical.items():
        expected = expected_grad[name]
        if expected is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, expected, rtol=2e-5, atol=2e-6)
            numerator += float((p.grad - expected).double().square().sum())
            denominator += float(expected.double().square().sum())
    organizer_norms = {}
    for tau, head in model.core.backend.organizer.control_heads.items():
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        organizer_norms[tau] = sum(float(g.double().square().sum()) for g in grads) ** .5
        assert organizer_norms[tau] > 0
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    optimizer.step()  # One disposable step; no checkpoint or scientific run.
    for name, value in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    for name, value in model.local_coupling.named_buffers():
        torch.testing.assert_close(value, local_buffers[name], rtol=0, atol=0)
    print({"local_shadow_native_loss": expected_value, "case_ids": case_ids, "queries": queries,
           "physical_gradient_relative_error": (numerator / max(denominator, 1e-300)) ** .5,
           "organizer_control_gradient_norms": organizer_norms, "native_work": snapshot})


@pytest.mark.skipif(not os.environ.get("HONF_TREE_SHADOW_CHECKPOINT"),
                    reason="Needs a fixed-quarter retained checkpoint and existing train atlas")
def test_native_callbacks_count_measured_wrappers_for_both_gradient_policies():
    from pathlib import Path

    from channelthermal.data.collation import ChannelThermalBatchCollator
    from channelthermal.data.datasets import GlobalChannelThermalDataset, H5Normalizer
    from channelthermal.data.development_split import development_case_ids
    from channelthermal.evaluation.loading import load_model
    from channelthermal.training.campaign_response import NativeCampaignResponse
    from channelthermal.training.epoch import make_model_inputs
    from torch.utils.data import DataLoader

    from honf_runtime.compat import recursive_to_device

    device = torch.device(os.environ.get("HONF_TREE_SHADOW_DEVICE", "cpu"))
    model, checkpoint = load_model(Path(os.environ["HONF_TREE_SHADOW_CHECKPOINT"]), device)
    cfg = checkpoint["train_config"]
    settings = cfg["dataset"]
    dataset = GlobalChannelThermalDataset(settings["packed_h5_path"], split="train", points_per_case=17,
        normalize_inputs=settings["normalize_inputs"], normalize_targets=settings["normalize_targets"],
        random_point_sampling=False, include_grid=False,
        normalizer=H5Normalizer(checkpoint["global_normalization_stats"]),
        case_ids=development_case_ids(settings["development_subset"], "train"))
    batch = recursive_to_device(next(iter(DataLoader(dataset, batch_size=1,
        collate_fn=ChannelThermalBatchCollator()))), device)
    inputs = make_model_inputs(batch, local_port_condition_mode="predicted", mixed_teacher_ratio=0.,
                               return_predicted_port_outputs=True, return_port_global_consistency=True)
    native_queries = None
    for policy, expected_atlas, expected_null in (("whole_wrapper_shadow_v1", 4, 8),
                                                 ("local_context_shadow_v1", 2, 4)):
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.train()
        model.set_training_progress(epoch=201, total_epochs=1000)
        model.zero_grad(set_to_none=True)
        model.campaign_training_state = deepcopy(checkpoint.get("campaign_training_state", {}))
        campaign = deepcopy(cfg["training"]["campaign"])
        campaign["organizer_gradient_policy"] = policy
        response = NativeCampaignResponse(model, dataset, settings, campaign)
        output = hard_value_soft_hypergraph_forward(model, gradient_policy=policy, **inputs)
        native = output["pred_field"].square().mean()
        auxiliary, metrics = response(201, native, 1.)
        assert metrics["response_wrapper_calls"] == expected_atlas
        assert metrics["heat_null_wrapper_calls"] == expected_null
        assert metrics["response_wrapper_calls"] == sum(
            phases["P0"]["prepare_calls"] for phases in metrics["response_forward_work"].values())
        assert metrics["heat_null_wrapper_calls"] == sum(
            phases["P0"]["prepare_calls"] for phases in metrics["heat_null_forward_work"].values())
        assert metrics["response_wrapper_count_basis"] == "measured native P0 preparations"
        assert metrics["heat_null_wrapper_count_basis"] == "measured native P0 preparations"
        queries = tuple(metrics[key] for key in
                        ("response_queries", "heat_null_primary_fluid_queries", "heat_null_role_queries"))
        if native_queries is None:
            native_queries = queries
        else:
            assert queries == native_queries
        (native + auxiliary).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
