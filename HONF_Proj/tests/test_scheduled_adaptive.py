from __future__ import annotations

import copy
import types

import torch

import honf_forward_core.organizer as organizer_module
import honf_forward_core.routing as routing_module
from honf_forward_core.config import BatchData, UnifiedForwardConfig
from honf_forward_core.decoder import HypergraphFieldDecoder, HypergraphGatedPairwiseKernel
from honf_forward_core.model import HONFNeuralField
from honf_forward_core.organizer import _scheduled_stabilized_assignment
from honf_forward_core.routing import entmax15, locality_bias, normalize_assignment, schedule_fraction


def _config(**overrides: object) -> UnifiedForwardConfig:
    values: dict[str, object] = {
        "field_dim": 3,
        "domain_length_x": 6.0,
        "domain_length_y": 3.0,
        "coordinate_scale": [6.0, 3.0],
        "periodic_axes": [],
        "num_env_tokens_x": 4,
        "num_env_tokens_y": 2,
        "num_hyperedges": 4,
        "organizer_mode": "exchangeable_slots",
        "edge_capacity": 4,
        "initial_active_edges": 4,
        "minimum_active_edges": 2,
        "slot_refinement_steps": 2,
        "edge_selection_mode": "quality_coverage",
        "selection_start_epoch": 150,
        "selection_transition_epochs": 250,
        "selection_warmup_mode": "all_viable",
        "selection_minimum_module_mass_fraction": 1.0e-6,
        "selection_minimum_environment_mass_fraction": 1.0e-6,
        "candidate_module_mass_fraction_floor": 1.0e-6,
        "candidate_environment_mass_fraction_floor": 1.0e-6,
        "module_assignment_normalizer": "scheduled",
        "module_sparsity_start_epoch": 350,
        "module_sparsity_transition_epochs": 300,
        "environment_assignment_normalizer": "scheduled",
        "environment_sparsity_start_epoch": 350,
        "environment_sparsity_transition_epochs": 300,
        "query_assignment_normalizer": "scheduled",
        "query_sparsity_start_epoch": 250,
        "query_sparsity_transition_epochs": 250,
        "environment_locality_mode": "gaussian_bounded",
        "environment_locality_strength": 0.25,
        "query_locality_mode": "none",
        "minimum_region_scale": 0.1,
        "hidden_dim": 24,
        "dropout": 0.0,
        "decoder_mode": "enhanced_honf_pairwise",
        "pairwise_kernel_hidden_dim": 24,
        "pairwise_kernel_num_layers": 2,
        "mechanism_state_mode": "descriptor_first",
        "field_assembly_mode": "edge_additive",
        "routing_execution": "scheduled",
        "gathered_execution_start_epoch": 650,
        "query_edge_limit": 3,
        "query_module_limit": 2,
        "query_edge_retained_mass_floor": 0.98,
        "module_incidence_retained_mass_floor": 0.95,
    }
    values.update(overrides)
    return UnifiedForwardConfig(**values)


def _batch() -> BatchData:
    generator = torch.Generator().manual_seed(701)
    return BatchData(
        module_centers=torch.rand(2, 5, 2, generator=generator) * torch.tensor([6.0, 3.0]),
        module_present=torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0]]),
        module_features=torch.randn(2, 5, 4, generator=generator),
        global_context=torch.randn(2, 3, generator=generator),
        query_xy=torch.rand(2, 9, 2, generator=generator) * torch.tensor([6.0, 3.0]),
        query_time=None,
        target_field=None,
        case_name="scheduled-adaptive-test",
        metadata={},
    )


def test_schedule_endpoints_and_intermediate_interpolation() -> None:
    assert schedule_fraction(149, 150, 250) == 0.0
    assert schedule_fraction(150, 150, 250) == 0.0
    assert schedule_fraction(275, 150, 250) == 0.5
    assert schedule_fraction(400, 150, 250) == 1.0
    assert schedule_fraction(900, -1, 250) == 0.0


def test_scheduled_probability_endpoints_and_intermediate_are_normalized() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0, -3.0]])
    soft = normalize_assignment(logits, mode="softmax")
    sparse = entmax15(logits)
    start = normalize_assignment(logits, mode="scheduled", entmax_blend=0.0)
    middle = normalize_assignment(logits, mode="scheduled", entmax_blend=0.4)
    end = normalize_assignment(logits, mode="scheduled", entmax_blend=1.0)

    torch.testing.assert_close(start, soft, rtol=0.0, atol=0.0)
    torch.testing.assert_close(end, sparse, rtol=0.0, atol=0.0)
    torch.testing.assert_close(middle, 0.6 * soft + 0.4 * sparse)
    torch.testing.assert_close(middle.sum(dim=-1), torch.ones(1))


def test_scheduled_entmax_endpoint_skips_softmax_branch(monkeypatch) -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0, -3.0]])
    original_softmax = torch.softmax
    calls = {"softmax": 0}

    def counted_softmax(*args, **kwargs):
        calls["softmax"] += 1
        return original_softmax(*args, **kwargs)

    monkeypatch.setattr(torch, "softmax", counted_softmax)
    result = routing_module.normalize_assignment(
        logits,
        mode="scheduled",
        entmax_blend=1.0,
    )
    assert calls["softmax"] == 0
    torch.testing.assert_close(result, entmax15(logits), rtol=0.0, atol=0.0)


def test_stabilized_entmax_endpoint_skips_stabilization(monkeypatch) -> None:
    logits = torch.tensor([[[5.0, 0.0, -5.0]]])

    def forbidden_stabilization(*args, **kwargs):
        raise AssertionError("softmax stabilization must not execute at mu=1")

    monkeypatch.setattr(
        organizer_module,
        "_stabilize_all_edge_softmax_assignment",
        forbidden_stabilization,
    )
    result = organizer_module._scheduled_stabilized_assignment(
        logits,
        entmax_blend=1.0,
        mass_fraction_floor=0.01,
    )
    torch.testing.assert_close(result, entmax15(logits), rtol=0.0, atol=0.0)


def test_scheduled_endpoints_are_finite_and_normalized_with_masks() -> None:
    logits = torch.tensor([[3.0, 1.0, -2.0, 7.0]])
    mask = torch.tensor([[True, True, True, False]])
    for blend in (0.0, 0.37, 1.0):
        assignment = normalize_assignment(
            logits,
            mode="scheduled",
            mask=mask,
            entmax_blend=blend,
        )
        assert torch.isfinite(assignment).all()
        assert assignment[0, -1] == 0
        torch.testing.assert_close(assignment.sum(dim=-1), torch.ones(1))

        organizer_logits = torch.stack([logits, logits], dim=1)
        token_mask = torch.tensor([[[1.0], [0.0]]])
        stabilized = _scheduled_stabilized_assignment(
            organizer_logits,
            entmax_blend=blend,
            mass_fraction_floor=0.01,
            mask=token_mask > 0,
            token_mask=token_mask,
        )
        assert torch.isfinite(stabilized).all()
        torch.testing.assert_close(stabilized[0, 0].sum(), torch.tensor(1.0))
        assert torch.count_nonzero(stabilized[0, 1]) == 0


def test_scheduled_candidate_assignment_blends_stabilized_softmax_into_exact_entmax() -> None:
    logits = torch.tensor(
        [
            [[12.0, 3.0, -2.0, -7.0], [5.0, -1.0, -4.0, -9.0]],
            [[8.0, 2.0, -3.0, -8.0], [4.0, 0.0, -5.0, -10.0]],
        ]
    )
    floor = 0.01
    raw_softmax = torch.softmax(logits, dim=-1)
    lower_bound = floor * 1.001
    expected_softmax = raw_softmax * (1.0 - logits.shape[-1] * lower_bound) + lower_bound
    expected_entmax = entmax15(logits)

    at_start = _scheduled_stabilized_assignment(
        logits,
        entmax_blend=0.0,
        mass_fraction_floor=floor,
    )
    in_transition = _scheduled_stabilized_assignment(
        logits,
        entmax_blend=0.4,
        mass_fraction_floor=floor,
    )
    at_end = _scheduled_stabilized_assignment(
        logits,
        entmax_blend=1.0,
        mass_fraction_floor=floor,
    )

    torch.testing.assert_close(at_start, expected_softmax, rtol=0.0, atol=0.0)
    torch.testing.assert_close(at_end, expected_entmax, rtol=0.0, atol=0.0)
    torch.testing.assert_close(in_transition, 0.6 * expected_softmax + 0.4 * expected_entmax)
    for assignment in (at_start, in_transition, at_end):
        torch.testing.assert_close(
            assignment.sum(dim=-1),
            torch.ones_like(assignment.sum(dim=-1)),
        )
    assert torch.all(at_start > floor)
    assert torch.count_nonzero(at_end == 0) > 0


def test_fresh_epoch_zero_scheduled_models_keep_all_eight_candidates_viable() -> None:
    config = _config(
        edge_capacity=8,
        initial_active_edges=8,
        candidate_module_mass_fraction_floor=0.01,
        candidate_environment_mass_fraction_floor=0.01,
        selection_minimum_module_mass_fraction=0.01,
        selection_minimum_environment_mass_fraction=0.01,
        routing_execution="dense",
    )
    batch = _batch()
    active_modules = batch.module_present > 0
    for seed in (11, 29, 47, 83):
        torch.manual_seed(seed)
        model = HONFNeuralField(config).eval()
        model.set_training_progress(epoch=0, total_epochs=1000)
        with torch.no_grad():
            output = model.encode_and_organize(batch)

        assert torch.equal(output["edge_viable_mask"], torch.ones_like(output["edge_viable_mask"]))
        assert torch.equal(output["edge_active_mask"], torch.ones_like(output["edge_active_mask"]))
        assert torch.all(output["candidate_module_mass_fraction"] > 0.01)
        assert torch.all(output["candidate_environment_mass_fraction"] > 0.01)
        torch.testing.assert_close(
            output["candidate_A_mh"].sum(dim=-1)[active_modules],
            torch.ones_like(output["candidate_A_mh"].sum(dim=-1)[active_modules]),
        )
        torch.testing.assert_close(
            output["candidate_A_eh"].sum(dim=-1),
            torch.ones_like(output["candidate_A_eh"].sum(dim=-1)),
        )


def test_selection_override_all_bypasses_greedy_selection(monkeypatch) -> None:
    model = HONFNeuralField(_config()).eval()
    model.set_training_progress(epoch=500, total_epochs=700)

    def fail_if_called(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("greedy selector must not run for selection_override='all'")

    monkeypatch.setattr(model.organizer.exchangeable, "_select_active_edges", fail_if_called)
    with torch.no_grad():
        output = model.encode_and_organize(
            _batch(),
            organizer_selection_override="all",
        )
    torch.testing.assert_close(output["edge_active_mask"], output["edge_viable_mask"])


def test_zero_selection_fraction_bypasses_greedy_selection(monkeypatch) -> None:
    model = HONFNeuralField(_config()).eval()
    model.set_training_progress(epoch=150, total_epochs=700)

    def fail_if_called(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("greedy selector must not run at selection fraction zero")

    monkeypatch.setattr(model.organizer.exchangeable, "_select_active_edges", fail_if_called)
    with torch.no_grad():
        output = model.encode_and_organize(_batch())
    assert output["selection_transition_fraction"].item() == 0.0
    torch.testing.assert_close(output["edge_active_mask"], output["edge_viable_mask"])


def test_edge_gate_is_continuous_and_exactly_hard_at_transition_end() -> None:
    torch.manual_seed(703)
    model = HONFNeuralField(_config()).eval()
    organizer = model.organizer.exchangeable

    def fixed_selection(self: object, *args: object, **kwargs: object) -> torch.Tensor:
        quality = args[2]
        mask = torch.zeros_like(quality)
        mask[:, :2] = 1.0
        return mask

    organizer._select_active_edges = types.MethodType(fixed_selection, organizer)
    outputs = {}
    for epoch in (149, 150, 275, 400):
        model.set_training_progress(epoch=epoch, total_epochs=700)
        with torch.no_grad():
            outputs[epoch] = model.encode_and_organize(_batch())

    torch.testing.assert_close(outputs[149]["edge_transition_gate"], torch.ones(2, 4))
    torch.testing.assert_close(outputs[150]["edge_transition_gate"], torch.ones(2, 4))
    torch.testing.assert_close(
        outputs[275]["edge_transition_gate"],
        torch.tensor([[1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 0.5, 0.5]]),
    )
    torch.testing.assert_close(outputs[400]["edge_transition_gate"], outputs[400]["hard_selected_edge_mask"])
    assert torch.equal(outputs[400]["edge_active_mask"], outputs[400]["hard_selected_edge_mask"])


def test_scheduled_selection_is_identical_in_train_and_eval_at_same_progress() -> None:
    torch.manual_seed(709)
    model = HONFNeuralField(_config())
    model.set_training_progress(epoch=275, total_epochs=700)
    batch = _batch()
    with torch.no_grad():
        model.train()
        train_output = model.encode_and_organize(batch)
        model.eval()
        eval_output = model.encode_and_organize(batch)
    for key in ("hard_selected_edge_mask", "edge_transition_gate", "A_mh", "A_eh"):
        torch.testing.assert_close(train_output[key], eval_output[key], rtol=0.0, atol=0.0)


def test_scheduled_progress_survives_checkpoint_round_trip() -> None:
    source = HONFNeuralField(_config())
    source.set_training_progress(epoch=431, total_epochs=700)
    with torch.no_grad():
        source(_batch())
    target = HONFNeuralField(_config())
    with torch.no_grad():
        target(_batch())
    target.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    assert target.selection_state() == {"epoch": 431, "total_epochs": 700}


def test_fresh_scheduled_model_has_explicit_final_inference_state() -> None:
    model = HONFNeuralField(_config())
    assert model.selection_state() == {"epoch": 650, "total_epochs": None}


def test_gaussian_bounded_locality_has_exact_cap() -> None:
    radius_square = torch.tensor([0.0, 1.0, 3.0, 20.0])
    bias = locality_bias(radius_square, mode="gaussian_bounded", strength=0.25, radius_cap=3.0)
    torch.testing.assert_close(bias, torch.tensor([0.0, -0.125, -0.375, -0.375]))


def test_query_route_limit_expands_only_to_retained_mass_floor() -> None:
    decoder = HypergraphFieldDecoder(_config(query_edge_limit=3, query_edge_retained_mass_floor=0.98))
    probabilities = torch.tensor([[[0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02]]])
    active = torch.ones(1, 8)
    limited, retained = decoder._limit_query_edge_routes(probabilities, active)
    assert int((limited > 0).sum()) > 3
    assert retained.item() >= 0.98 - 1.0e-6
    torch.testing.assert_close(limited.sum(dim=-1), torch.ones(1, 1))

    concentrated = torch.tensor([[[0.70, 0.20, 0.09, 0.01, 0.0, 0.0, 0.0, 0.0]]])
    limited_concentrated, retained_concentrated = decoder._limit_query_edge_routes(concentrated, active)
    assert int((limited_concentrated > 0).sum()) == 3
    torch.testing.assert_close(retained_concentrated, torch.tensor([[0.99]]))


def test_module_route_limit_expands_and_preserves_normalized_scale() -> None:
    config = _config(query_module_limit=1, module_incidence_retained_mass_floor=0.95)
    kernel = HypergraphGatedPairwiseKernel(config)

    class Ones(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.ones(*values.shape[:-1], config.hidden_dim, device=values.device, dtype=values.dtype)

    kernel.pair_mlp = Ones()
    contexts, selected, _, retained = kernel._gathered_edge_pair_context(
        torch.tensor([[[0.0, 0.0]]]),
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]),
        torch.zeros(1, 3, config.hidden_dim),
        torch.ones(1, 3),
        torch.tensor([[[0.60], [0.30], [0.10]]]),
        torch.ones(1, 1, 1),
        None,
    )
    assert selected.item() == 3.0
    assert retained.item() >= 0.95
    torch.testing.assert_close(contexts, torch.ones_like(contexts))
