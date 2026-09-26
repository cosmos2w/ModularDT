"""Focused CUDA mechanism tests for anchored response-factor HONF."""

from __future__ import annotations

import inspect
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from honf_forward_core.interaction_response import (
    DEFAULT_OUTPUT_CHANNELS,
    AnchoredResponseFactorOperator,
    InputOnlySupportScorer,
    ReceiverSupport,
    ResponseFactor,
    ResponseQueryBatch,
    ValidityNeighborhood,
    enumerate_unary_pair_candidates,
    group_sparse_support_search,
    make_baseline_cache,
    masked_role_response_loss,
    masked_support_classification_loss,
    response_factor_training_step,
    select_factor_control,
    size_matched_random_support,
    stencil_examples_to_torch,
)


def _cuda2() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        pytest.skip("interaction-response mechanism checks require physical GPU 2")
    return torch.device("cuda:2")


def _cache(module_ids: tuple[str, ...], *, device: torch.device, seed: int = 13):
    generator = torch.Generator(device=device).manual_seed(seed)
    count = len(module_ids)
    return make_baseline_cache(
        module_ids,
        torch.randn(1, count, 4, generator=generator, device=device),
        torch.randn(1, count, 3, generator=generator, device=device),
        torch.randn(1, 5, generator=generator, device=device),
    )


def _queries(module_ids: tuple[str, ...], *, device: torch.device, seed: int = 29):
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        "fluid_fields": ResponseQueryBatch(
            features=torch.randn(1, 7, 2, generator=generator, device=device),
            query_ids=tuple(f"fluid:{index}" for index in range(7)),
        ),
        "interface": ResponseQueryBatch(
            features=torch.randn(1, 8, 3, generator=generator, device=device),
            receiver_module_ids=tuple(module_ids[index % len(module_ids)] for index in range(8)),
            query_ids=tuple((module_ids[index % len(module_ids)], index // len(module_ids)) for index in range(8)),
        ),
        "solid_temperature": ResponseQueryBatch(
            features=torch.randn(1, 9, 2, generator=generator, device=device),
            receiver_module_ids=tuple(module_ids[index % len(module_ids)] for index in range(9)),
            query_ids=tuple((module_ids[index % len(module_ids)], "solid", index) for index in range(9)),
        ),
    }


def _operator(
    device: torch.device,
    *,
    hidden_dim: int = 24,
    decoder_mode: str = "inclusion_exclusion",
) -> AnchoredResponseFactorOperator:
    return AnchoredResponseFactorOperator(
        module_feature_dim=4,
        design_dim=3,
        context_dim=5,
        delta_dim=3,
        hidden_dim=hidden_dim,
        decoder_mode=decoder_mode,
    ).to(device)


def _factor(donors: tuple[str, ...], *, factor_id: str = "factor") -> ResponseFactor:
    return ResponseFactor(
        factor_id=factor_id,
        donor_ids=donors,
        output_roles=tuple(DEFAULT_OUTPUT_CHANNELS),
        validity=ValidityNeighborhood(
            max_abs_delta_by_module={module_id: (0.25, 0.25, 0.5) for module_id in donors},
            baseline_design_center_by_module={module_id: (0.0, 0.0, 1.0) for module_id in donors},
            context_radius=2.0,
            context_center=(0.0, 0.0, 0.0, 0.0, 0.0),
            anchor_family_ids=("synthetic-anchor",),
        ),
        evidence_sources=("analytic_synthetic",),
        evidence_anchor_ids=("synthetic-anchor",),
    )


def _deltas(module_ids: tuple[str, ...], *, device: torch.device, seed: int = 47):
    generator = torch.Generator(device=device).manual_seed(seed)
    return {module_id: torch.randn(1, 3, generator=generator, device=device) * 0.05 for module_id in module_ids}


def test_anchored_unary_pair_zero_and_delta_derivative() -> None:
    device = _cuda2()
    ids = ("module-a", "module-b", "module-c")
    cache = _cache(ids, device=device)
    queries = _queries(ids, device=device)
    model = _operator(device)
    unary = _factor((ids[0],), factor_id="unary")
    pair = _factor((ids[0], ids[2]), factor_id="pair")
    delta = _deltas(ids, device=device)

    zero_unary = dict(delta)
    zero_unary[ids[0]] = torch.zeros_like(delta[ids[0]])
    unary_output = model.predict_factor_response(cache, unary, zero_unary, queries)
    for role, values in unary_output.items():
        assert torch.equal(values, torch.zeros_like(values)), role

    zero_pair = dict(delta)
    zero_pair[ids[2]] = torch.zeros_like(delta[ids[2]])
    pair_output = model.predict_factor_response(cache, pair, zero_pair, queries)
    for role, values in pair_output.items():
        assert torch.equal(values, torch.zeros_like(values)), role

    variable = delta[ids[0]].detach().clone().requires_grad_(True)
    derivative_inputs = dict(delta)
    derivative_inputs[ids[0]] = variable
    value = model.predict_factor_response(cache, unary, derivative_inputs, queries)["fluid_fields"].square().sum()
    derivative = torch.autograd.grad(value, variable)[0]
    direction = torch.tensor([[0.3, -0.2, 0.4]], device=device)
    step = 1.0e-3
    plus = dict(derivative_inputs)
    minus = dict(derivative_inputs)
    plus[ids[0]] = variable.detach() + step * direction
    minus[ids[0]] = variable.detach() - step * direction
    fp = model.predict_factor_response(cache, unary, plus, queries)["fluid_fields"].square().sum()
    fm = model.predict_factor_response(cache, unary, minus, queries)["fluid_fields"].square().sum()
    finite = (fp - fm) / (2.0 * step)
    automatic = (derivative * direction).sum()
    assert torch.isfinite(derivative).all()
    assert torch.allclose(automatic, finite, rtol=2.0e-2, atol=2.0e-3)


def test_delta_gated_decoder_preserves_anchor_pair_cross_and_query_order() -> None:
    device = _cuda2()
    ids = ("a", "b", "c")
    cache = _cache(ids, device=device)
    queries = _queries(ids, device=device)
    model = _operator(device, decoder_mode="delta_gated")
    scales = {
        "fluid_fields": (2.0, 1.0, 0.5, 3.0, 4.0),
        "interface": (2.0, 3.0),
        "solid_temperature": (5.0,),
    }
    model.set_response_scales(scales)
    unary = _factor(("a",), factor_id="unary-a")
    pair = _factor(("a", "b"), factor_id="pair-a-b")
    reverse_pair = _factor(("b", "a"), factor_id="pair-b-a")
    deltas = _deltas(ids, device=device, seed=107)

    zero = dict(deltas)
    zero["a"] = torch.zeros_like(zero["a"])
    zero_unary = model.predict_factor_response(cache, unary, zero, queries)
    assert all(torch.equal(value, torch.zeros_like(value)) for value in zero_unary.values())

    pair_output = model.predict_factor_response(cache, pair, deltas, queries, query_chunk_size=3)
    reverse_output = model.predict_factor_response(cache, reverse_pair, deltas, queries, query_chunk_size=3)
    assert all(torch.allclose(pair_output[role], reverse_output[role], atol=1.0e-6, rtol=1.0e-6) for role in pair_output)
    zero_first = dict(deltas)
    zero_first["a"] = torch.zeros_like(zero_first["a"])
    zero_second = dict(deltas)
    zero_second["b"] = torch.zeros_like(zero_second["b"])
    for values in (
        model.predict_factor_response(cache, pair, zero_first, queries),
        model.predict_factor_response(cache, pair, zero_second, queries),
    ):
        assert all(torch.equal(value, torch.zeros_like(value)) for value in values.values())

    variable = torch.zeros(1, 3, device=device, requires_grad=True)
    at_zero = dict(deltas)
    at_zero["a"] = variable
    unary_sum = model.predict_factor_response(cache, unary, at_zero, queries)["fluid_fields"].sum()
    gradient = torch.autograd.grad(unary_sum, variable)[0]
    assert torch.isfinite(gradient).all()
    assert bool(torch.count_nonzero(gradient))

    chunked = model.predict_factor_response(cache, pair, deltas, queries, query_chunk_size=1)
    assert all(torch.allclose(pair_output[role], chunked[role], atol=1.0e-6, rtol=1.0e-6) for role in pair_output)
    rescaled = dict(scales)
    rescaled["fluid_fields"] = tuple(2.0 * value for value in scales["fluid_fields"])
    model.set_response_scales(rescaled)
    double_fluid = model.predict_factor_response(cache, pair, deltas, queries, query_chunk_size=3)
    assert torch.allclose(double_fluid["fluid_fields"], 2.0 * pair_output["fluid_fields"], atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(double_fluid["interface"], pair_output["interface"], atol=1.0e-6, rtol=1.0e-6)


def test_factor_donor_exclusion_and_no_trial_response_argument() -> None:
    device = _cuda2()
    ids = ("a", "b", "c")
    cache = _cache(ids, device=device)
    queries = _queries(ids, device=device)
    model = _operator(device)
    factor = _factor(("a", "c"), factor_id="a-c")
    deltas = _deltas(ids, device=device)
    first = model.predict_factor_response(cache, factor, deltas, queries)
    changed = dict(deltas)
    changed["b"] = torch.full_like(changed["b"], 50.0)
    second = model.predict_factor_response(cache, factor, changed, queries)
    assert all(torch.equal(first[role], second[role]) for role in first)
    assert "trial_design" not in inspect.signature(model.predict_factor_response).parameters
    assert "trial_field" not in inspect.signature(model.predict_factor_response).parameters

    outside = torch.ones(1, 3, device=device, requires_grad=True)
    changed["b"] = outside
    output = model.predict_factor_response(cache, factor, changed, queries)
    torch.autograd.grad(sum(value.square().sum() for value in output.values()), outside, allow_unused=True)
    assert outside.grad is None


def test_module_permutation_query_chunks_and_stable_receiver_support() -> None:
    device = _cuda2()
    ids = ("a", "b", "c", "d")
    cache = _cache(ids, device=device)
    queries = _queries(ids, device=device)
    model = _operator(device)
    factor = _factor(("a", "d"), factor_id="a-d")
    deltas = _deltas(ids, device=device)
    expected = model.predict_factor_response(cache, factor, deltas, queries)
    chunked = model.predict_factor_response(cache, factor, deltas, queries, query_chunk_size=2)
    assert all(torch.allclose(expected[role], chunked[role], atol=1.0e-6, rtol=1.0e-6) for role in expected)

    reordered_queries = {}
    query_permutations = {}
    for role, query in queries.items():
        permutation_q = torch.randperm(query.features.shape[1], device=device)
        positions = permutation_q.cpu().tolist()
        query_permutations[role] = permutation_q
        reordered_queries[role] = ResponseQueryBatch(
            features=query.features.index_select(1, permutation_q),
            receiver_module_ids=(
                None
                if query.receiver_module_ids is None
                else tuple(query.receiver_module_ids[index] for index in positions)
            ),
            query_ids=None if query.query_ids is None else tuple(query.query_ids[index] for index in positions),
            mask=None if query.mask is None else query.mask.index_select(1, permutation_q),
            quadrature_weights=(
                None if query.quadrature_weights is None else query.quadrature_weights.index_select(1, permutation_q)
            ),
        )
    reordered_output = model.predict_factor_response(cache, factor, deltas, reordered_queries)
    assert all(
        torch.allclose(
            expected[role].index_select(1, query_permutations[role]),
            reordered_output[role],
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        for role in expected
    )

    permutation = torch.tensor([2, 0, 3, 1], device=device)
    permuted_cache = make_baseline_cache(
        tuple(ids[index] for index in permutation.cpu().tolist()),
        cache.module_features.index_select(1, permutation),
        cache.baseline_design.index_select(1, permutation),
        cache.baseline_context,
    )
    permuted = model.predict_factor_response(permuted_cache, factor, deltas, queries)
    assert all(torch.allclose(expected[role], permuted[role], atol=1.0e-6, rtol=1.0e-6) for role in expected)

    restricted = ResponseFactor(
        factor_id="restricted",
        donor_ids=("a", "d"),
        output_roles=tuple(DEFAULT_OUTPUT_CHANNELS),
        validity=factor.validity,
        receiver_support=ReceiverSupport(
            all_receivers=False,
            receiver_query_ids_by_role={"fluid_fields": ("fluid:1", "fluid:4")},
            receiver_module_ids_by_role={"interface": ("a",), "solid_temperature": ("d",)},
        ),
        evidence_sources=("stored_reference",),
    )
    localized = model.predict_factor_response(cache, restricted, deltas, queries, query_chunk_size=3)
    assert torch.equal(
        localized["fluid_fields"][:, [0, 2, 3, 5, 6]], torch.zeros_like(localized["fluid_fields"][:, [0, 2, 3, 5, 6]])
    )
    assert torch.equal(
        localized["interface"][
            :, [index for index, module in enumerate(queries["interface"].receiver_module_ids) if module != "a"]
        ],
        torch.zeros_like(
            localized["interface"][
                :, [index for index, module in enumerate(queries["interface"].receiver_module_ids) if module != "a"]
            ]
        ),
    )
    assert torch.equal(
        localized["solid_temperature"][
            :, [index for index, module in enumerate(queries["solid_temperature"].receiver_module_ids) if module != "d"]
        ],
        torch.zeros_like(
            localized["solid_temperature"][
                :,
                [
                    index
                    for index, module in enumerate(queries["solid_temperature"].receiver_module_ids)
                    if module != "d"
                ],
            ]
        ),
    )


def test_empty_and_inactive_graphs_skip_factor_decoding() -> None:
    device = _cuda2()
    ids = ("a", "b", "c")
    cache = _cache(ids, device=device)
    queries = _queries(ids, device=device)
    model = _operator(device)
    factor = _factor(("a", "b"), factor_id="inactive")
    deltas = _deltas(ids, device=device)

    empty = model.predict_response(cache, (), deltas, queries)
    assert all(torch.equal(value, torch.zeros_like(value)) for value in empty.values())
    skipped, skipped_ids = model.predict_response(
        cache,
        (factor,),
        deltas,
        queries,
        factor_weights={factor.factor_id: torch.zeros(1, device=device)},
        return_evaluated_factor_ids=True,
    )
    padded, padded_ids = model.predict_response(
        cache,
        (factor,),
        deltas,
        queries,
        factor_weights={factor.factor_id: 0.0},
        skip_inactive=False,
        return_evaluated_factor_ids=True,
    )
    assert skipped_ids == ()
    assert padded_ids == (factor.factor_id,)
    assert all(torch.equal(skipped[role], padded[role]) for role in skipped)


def test_validity_neighborhood_is_explicit_and_bounded() -> None:
    device = _cuda2()
    ids = ("a", "b", "c")
    cache = _cache(ids, device=device)
    model = _operator(device)
    deltas = _deltas(ids, device=device)
    centers = {
        module_id: tuple(float(value) for value in cache.baseline_design[0, index].cpu().tolist())
        for index, module_id in enumerate(ids)
    }
    factor = ResponseFactor(
        factor_id="bounded-a-c",
        donor_ids=("a", "c"),
        output_roles=("fluid_fields",),
        validity=ValidityNeighborhood(
            max_abs_delta_by_module={"a": (0.2, 0.2, 0.5), "c": (0.2, 0.2, 0.5)},
            baseline_design_center_by_module=centers,
            context_radius=1.0,
            context_center=tuple(float(value) for value in cache.baseline_context[0].cpu().tolist()),
        ),
    )
    assert model.validity_status(cache, factor, deltas) is True
    outside = dict(deltas)
    outside["a"] = torch.full((1, 3), 0.3, device=device)
    assert model.validity_status(cache, factor, outside) is False
    unknown = ResponseFactor(
        factor_id="unknown-neighborhood",
        donor_ids=("a", "c"),
        output_roles=("fluid_fields",),
        validity=ValidityNeighborhood(max_abs_delta_by_module={}),
    )
    assert model.validity_status(cache, unknown, deltas) is None

    partially_measured = ResponseFactor(
        factor_id="horizontal-only-neighborhood",
        donor_ids=("a",),
        output_roles=("fluid_fields",),
        validity=ValidityNeighborhood(
            max_abs_delta_by_module={"a": (0.2, None, None)},
            baseline_design_center_by_module={"a": centers["a"]},
            context_radius=1.0,
            context_center=tuple(float(value) for value in cache.baseline_context[0].cpu().tolist()),
        ),
    )
    within_horizontal = dict(deltas)
    within_horizontal["a"] = torch.tensor([[0.1, 0.1, 0.0]], device=device)
    assert model.validity_status(cache, partially_measured, within_horizontal) is None
    outside_horizontal = dict(within_horizontal)
    outside_horizontal["a"] = torch.tensor([[0.3, 0.0, 0.0]], device=device)
    assert model.validity_status(cache, partially_measured, outside_horizontal) is False


def test_input_only_support_scorer_and_masked_labels() -> None:
    device = _cuda2()
    ids = ("a", "b", "c", "d")
    cache = _cache(ids, device=device)
    factors = enumerate_unary_pair_candidates(ids, tuple(DEFAULT_OUTPUT_CHANNELS))
    scorer = InputOnlySupportScorer(module_feature_dim=4, design_dim=3, context_dim=5, hidden_dim=16).to(device)
    scores = scorer.score_factors(cache, factors)
    assert "delta_by_module_id" not in inspect.signature(scorer.score_factors).parameters
    assert "response_targets" not in inspect.signature(scorer.score_factors).parameters
    assert scores.shape == (1, len(factors))
    weights = scorer.retention_weights(scores, threshold=0.0)
    assert torch.equal(weights[scores <= 0.0], torch.zeros_like(weights[scores <= 0.0]))
    assert bool((weights[scores > 0.0] > 0.0).all())

    target = torch.zeros_like(scores)
    resolved = torch.zeros_like(scores, dtype=torch.bool)
    target[:, 0] = 1.0
    target[:, 1] = float("nan")  # unresolved values cannot become negative labels
    resolved[:, 0] = True
    resolved[:, 1] = False
    loss = masked_support_classification_loss(scores, target, resolved)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in scorer.parameters())


def test_role_balanced_finite_response_loss_masks_unresolved_values() -> None:
    device = _cuda2()
    prediction = {
        "fluid_fields": torch.zeros(1, 256, 5, device=device, requires_grad=True),
        "interface": torch.zeros(1, 4, 2, device=device, requires_grad=True),
    }
    target = {
        "fluid_fields": torch.ones_like(prediction["fluid_fields"]),
        "interface": torch.full_like(prediction["interface"], 3.0),
    }
    fluid_mask = torch.ones(1, 256, dtype=torch.bool, device=device)
    fluid_mask[:, -1] = False
    target["fluid_fields"][:, -1] = float("nan")
    result, components = masked_role_response_loss(
        prediction,
        target,
        {"fluid_fields": fluid_mask, "interface": torch.ones(1, 4, dtype=torch.bool, device=device)},
        training_scales={"fluid_fields": 1.0, "interface": 3.0},
        return_components=True,
    )
    assert torch.allclose(result, torch.ones_like(result))
    assert torch.allclose(components["fluid_fields"], torch.ones_like(result))
    assert torch.allclose(components["interface"], torch.ones_like(result))
    result.backward()
    assert torch.isfinite(prediction["fluid_fields"].grad).all()


def test_stencil_conversion_and_reusable_optimizer_step() -> None:
    device = _cuda2()
    ids = ("a", "b", "c")
    cache = _cache(ids, device=device)
    queries = {
        "fluid_fields": ResponseQueryBatch(
            features=torch.randn(1, 7, 2, device=device),
            query_ids=tuple(f"q{index}" for index in range(7)),
        )
    }
    block = SimpleNamespace(
        delta=np.ones((7, 5), dtype=np.float64),
        valid_mask=np.ones((7, 5), dtype=bool),
        quadrature_weights=np.full(7, 0.25, dtype=np.float64),
    )
    raw_example = SimpleNamespace(
        record_id="physical-single-1",
        anchor_id="anchor-1",
        physical_family_id="family-1",
        split="train",
        source="reference_solver",
        perturbation_by_module={
            "a": (0.2, 0.0, 0.0),
            "b": (0.0, -0.4, 0.0),
            "c": (0.0, 0.0, 0.0),
        },
        roles={"fluid_fields": block},
    )
    example = stencil_examples_to_torch(
        (raw_example,),
        module_ids=ids,
        input_coordinate_scales=(2.0, 4.0, 1.0),
        device=device,
    )[0]
    assert example.source == "reference_solver"
    assert example.physical_family_id == "family-1"
    assert torch.allclose(example.delta_by_module_id["a"], torch.tensor([[0.1, 0.0, 0.0]], device=device))
    assert torch.equal(example.observed_masks["fluid_fields"], torch.ones(1, 7, 5, dtype=torch.bool, device=device))
    assert torch.equal(example.loss_masks["fluid_fields"], example.observed_masks["fluid_fields"])
    assert torch.equal(example.quadrature_weights["fluid_fields"], torch.full((1, 7), 0.25, device=device))

    model = _operator(device, hidden_dim=16)
    factor = _factor(("a",), factor_id="unary-a")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    before = model.output_heads["fluid_fields"].weight.detach().clone()
    result = response_factor_training_step(
        model,
        optimizer,
        cache,
        (factor,),
        queries,
        example,
        training_scales={"fluid_fields": 1.0},
        query_chunk_size=3,
    )
    assert result["response_kind"] == "finite_change"
    assert np.isfinite(float(result["loss"]))
    assert not torch.equal(before, model.output_heads["fluid_fields"].weight)


def test_stencil_observation_loss_and_sign_masks_are_distinct() -> None:
    device = _cuda2()
    targets = np.ones((3, 5), dtype=np.float64)
    block = SimpleNamespace(
        delta=targets,
        valid_mask=np.ones((3, 5), dtype=bool),
        quadrature_weights=np.ones(3, dtype=np.float64),
    )
    raw_example = SimpleNamespace(
        record_id="mixed-corner-pp",
        anchor_id="anchor-1",
        physical_family_id="family-1",
        split="train",
        source="reference_solver",
        perturbation_by_module={"a": (0.2, 0.0, 0.0), "b": (0.0, -0.4, 0.0)},
        roles={"fluid_fields": block},
    )
    loss_mask = np.ones_like(targets, dtype=bool)
    loss_mask[:, 0] = False
    example = stencil_examples_to_torch(
        (raw_example,),
        module_ids=("a", "b"),
        input_coordinate_scales=(2.0, 4.0, 1.0),
        device=device,
        response_kind="anchored_mixed_response",
        loss_mask_overrides_by_record_role={("mixed-corner-pp", "fluid_fields"): loss_mask},
        sign_resolution_floors_by_role={"fluid_fields": (0.5, 1.5, 0.5, 0.5, 0.5)},
        mask_basis="common_stencil_geometry; measured noise floor for sign labels",
    )[0]
    assert example.mask_basis.startswith("common_stencil_geometry")
    assert example.response_kind == "anchored_mixed_response"
    assert bool(example.observed_masks["fluid_fields"].all())
    assert not bool(example.loss_masks["fluid_fields"][:, :, 0].any())
    assert bool(example.loss_masks["fluid_fields"][:, :, 1:].all())
    assert bool(example.sign_resolved_masks["fluid_fields"][:, :, 0].all())
    assert not bool(example.sign_resolved_masks["fluid_fields"][:, :, 1].any())


def test_train_only_group_support_search_and_matched_controls() -> None:
    device = _cuda2()
    factors = enumerate_unary_pair_candidates(("a", "b"), ("fluid_fields", "interface"))
    target_fluid = torch.arange(20, device=device, dtype=torch.float32).reshape(4, 5) / 10.0
    target_interface = torch.arange(8, device=device, dtype=torch.float32).reshape(4, 2) / 5.0
    fluid_predictions = torch.zeros(len(factors), 4, 5, device=device)
    interface_predictions = torch.zeros(len(factors), 4, 2, device=device)
    pair_index = next(index for index, factor in enumerate(factors) if len(factor.donor_ids) == 2)
    fluid_predictions[pair_index] = target_fluid
    interface_predictions[pair_index] = target_interface
    fluid_mask = torch.ones_like(target_fluid, dtype=torch.bool)
    interface_mask = torch.ones_like(target_interface, dtype=torch.bool)
    fluid_mask[-1, -1] = False
    interface_mask[-1, -1] = False
    target_fluid[-1, -1] = float("nan")
    target_interface[-1, -1] = float("nan")

    result = group_sparse_support_search(
        factors,
        {"fluid_fields": fluid_predictions, "interface": interface_predictions},
        {"fluid_fields": target_fluid, "interface": target_interface},
        {"fluid_fields": fluid_mask, "interface": interface_mask},
        response_scales={"fluid_fields": 1.0, "interface": 1.0},
        tolerances={"fluid_fields": 1.0e-5, "interface": 1.0e-5},
        max_additions=2,
    )
    assert result.adequate
    assert result.selected_factor_ids == (factors[pair_index].factor_id,)
    assert pair_index in [
        factor_index for factor_index, factor in enumerate(factors) if factor.factor_id in result.selected_factor_ids
    ]
    assert len(result.evaluated_candidate_ids) == len(factors)

    adaptive = select_factor_control(factors, "adaptive", adaptive_ids=result.selected_factor_ids)
    fixed = select_factor_control(factors, "fixed", fixed_ids=result.selected_factor_ids)
    random_a = size_matched_random_support(factors, adaptive, seed=17)
    random_b = size_matched_random_support(factors, adaptive, seed=17)
    assert len(adaptive) == len(fixed) == len(random_a) == 1
    assert tuple(value.factor_id for value in random_a) == tuple(value.factor_id for value in random_b)
    assert len(select_factor_control(factors, "unary_only")) == 2
    assert len(select_factor_control(factors, "all_candidates")) == len(factors)


@pytest.mark.parametrize("module_count", [3, 10])
def test_low_and_high_module_response_optimizer_step(module_count: int, record_property) -> None:
    """Exercise the response decoder and support scorer on CUDA 2 at two M."""

    device = _cuda2()
    ids = tuple(f"m{index}" for index in range(module_count))
    cache = _cache(ids, device=device, seed=module_count)
    queries = {
        "fluid_fields": ResponseQueryBatch(
            features=torch.randn(1, 11, 2, device=device),
            query_ids=tuple(f"q{index}" for index in range(11)),
        )
    }
    model = _operator(device, hidden_dim=16)
    candidates = enumerate_unary_pair_candidates(ids, ("fluid_fields",))
    scorer = InputOnlySupportScorer(
        module_feature_dim=4, design_dim=3, context_dim=5, role_names=("fluid_fields",), hidden_dim=16
    ).to(device)
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    step_started = time.perf_counter()
    scores = scorer.score_factors(cache, candidates)
    active_indices = (0, module_count)
    active_factors = tuple(candidates[index] for index in active_indices)
    support_weights = scorer.retention_weights(scores[:, list(active_indices)], threshold=-2.0)
    factor_weights = {
        factor.factor_id: support_weights[:, local_index] for local_index, factor in enumerate(active_factors)
    }
    deltas = _deltas(ids, device=device, seed=70 + module_count)
    prediction = model.predict_response(
        cache,
        active_factors,
        deltas,
        queries,
        factor_weights=factor_weights,
        query_chunk_size=4,
    )["fluid_fields"]
    target = torch.randn_like(prediction)
    support_target = torch.zeros_like(scores)
    support_target[:, list(active_indices)] = 1.0
    support_mask = torch.zeros_like(scores, dtype=torch.bool)
    support_mask[:, list(active_indices)] = True
    response_loss = masked_role_response_loss(
        {"fluid_fields": prediction},
        {"fluid_fields": target},
        {"fluid_fields": torch.ones_like(target[..., 0], dtype=torch.bool)},
        training_scales={"fluid_fields": 1.0},
    )
    loss = response_loss + masked_support_classification_loss(scores, support_target, support_mask)
    optimizer = torch.optim.Adam((*model.parameters(), *scorer.parameters()), lr=1.0e-3)
    before = model.output_heads["fluid_fields"].weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(parameter.grad.abs().sum() > 0.0)
        for parameter in model.parameters()
    )
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(parameter.grad.abs().sum() > 0.0)
        for parameter in scorer.parameters()
    )
    optimizer.step()
    torch.cuda.synchronize(device)
    step_elapsed = time.perf_counter() - step_started
    peak_increment = max(0, torch.cuda.max_memory_allocated(device) - allocated_before)
    record_property("module_count", module_count)
    record_property("device", str(device))
    record_property("finite_loss", float(loss.detach().item()))
    record_property("response_loss", float(response_loss.detach().item()))
    record_property("support_loss", float((loss - response_loss).detach().item()))
    record_property(
        "model_gradient_l1",
        float(sum(parameter.grad.abs().sum() for parameter in model.parameters() if parameter.grad is not None).item()),
    )
    record_property(
        "scorer_gradient_l1",
        float(
            sum(parameter.grad.abs().sum() for parameter in scorer.parameters() if parameter.grad is not None).item()
        ),
    )
    record_property(
        "output_head_update_l1", float((model.output_heads["fluid_fields"].weight - before).abs().sum().item())
    )
    record_property("step_wall_seconds", step_elapsed)
    record_property("step_peak_memory_increment_bytes", peak_increment)
    assert not torch.equal(before, model.output_heads["fluid_fields"].weight)
