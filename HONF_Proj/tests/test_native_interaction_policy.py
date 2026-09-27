"""Native Dense policy opt-in, exact all-access, and typed transport coverage."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.adaptive_cover_field import AdaptiveCoverPairwiseField
from honf_forward_core.interface_fields.adaptive_interaction_cover import AdaptiveCoverPlan
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField


def _config() -> UnifiedForwardConfig:
    return UnifiedForwardConfig(
        forward_architecture="dense_pairwise_field",
        field_dim=3,
        spatial_dim=3,
        coordinate_scale=[50.0, 38.0, 6.25],
        hidden_dim=16,
        dropout=0.0,
        geometry_mode="nonperiodic",
        boundary_feature_mode="none",
        interface_model=InterfaceFieldConfig(
            message_hidden_dim=8,
            attention_heads=2,
            coarse_latent_count=3,
            coarse_blocks=1,
            relative_fourier_frequencies=2,
            receiver_chunk_size=2,
        ),
    )


def _batch() -> BatchData:
    generator = torch.Generator().manual_seed(831)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    return BatchData(
        module_centers=random(1, 4, 3),
        module_present=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        module_features=random(1, 4, 3),
        global_context=random(1, 5),
        query_xy=random(1, 7, 3),
        query_time=None,
        target_field=random(1, 7, 3),
        case_name="native-policy-test",
        metadata={"review_split": "must_not_reach_policy"},
        env_coords=random(1, 8, 3),
        env_features=random(1, 8, 2),
        query_features=random(1, 7, 2),
        env_weights=torch.rand(1, 8, generator=generator) + 0.2,
    )


class _AllAccessPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prepared_keys: set[str] | None = None
        self.encoded_has_target = True

    def plan_cases(self, encoded, prepared_state, trees):
        self.prepared_keys = set(prepared_state)
        self.encoded_has_target = hasattr(encoded, "target_field")
        return tuple(
            AdaptiveCoverPlan.full_access(tree, encoded.module_present[case], int(encoded.env_coords.shape[1]))
            for case, tree in enumerate(trees)
        )


class _PrunedRootPolicy(nn.Module):
    def plan_cases(self, encoded, prepared_state, trees):
        assert set(prepared_state) == {"module_states", "environment_states", "global_state"}
        plans = []
        for case, tree in enumerate(trees):
            plan = AdaptiveCoverPlan.full_access(
                tree, encoded.module_present[case], int(encoded.env_coords.shape[1])
            )
            module_membership = plan.module_membership.clone()
            environment_membership = plan.environment_membership.clone()
            module_membership[0, 1] = 0.0
            environment_membership[0, 2] = 0.0
            plans.append(replace(
                plan,
                module_membership=module_membership,
                environment_membership=environment_membership,
            ))
        return tuple(plans)


def _dense_reference(config: UnifiedForwardConfig) -> InterfaceFieldCore:
    core = InterfaceFieldCore(config)
    options = config.interface_model
    assert options is not None
    core.backend = DensePairwiseField(
        int(config.hidden_dim),
        int(options.message_hidden_dim),
        int(options.attention_heads),
        int(options.relative_fourier_frequencies),
        activation_checkpointing=bool(options.activation_checkpointing),
    )
    return core


def _predict(core: InterfaceFieldCore, batch: BatchData) -> torch.Tensor:
    encoded = core.encode_case(batch)
    prepared = core.prepare(encoded, encoded.module_tokens)
    return core.decode_queries(
        prepared,
        batch.query_xy,
        batch.query_features,
        receiver_chunk_size=2,
    )["pred_field"]


def _parameter_and_input_gradients(core: InterfaceFieldCore, batch: BatchData):
    centers = batch.module_centers.detach().clone().requires_grad_(True)
    module_features = batch.module_features.detach().clone().requires_grad_(True)
    environment_features = batch.env_features.detach().clone().requires_grad_(True)
    queries = batch.query_xy.detach().clone().requires_grad_(True)
    current = replace(
        batch,
        module_centers=centers,
        module_features=module_features,
        env_features=environment_features,
        query_xy=queries,
    )
    prediction = _predict(core, current)
    named_parameters = dict(core.named_parameters())
    gradients = torch.autograd.grad(
        prediction.square().sum(),
        (*named_parameters.values(), centers, module_features, environment_features, queries),
        allow_unused=True,
    )
    return prediction, named_parameters, gradients


def _materialize(core: InterfaceFieldCore, batch: BatchData) -> None:
    with torch.no_grad():
        _predict(core, batch)


def _copy_reference_parameters(reference: InterfaceFieldCore, candidate: InterfaceFieldCore, batch: BatchData) -> None:
    _materialize(reference, batch)
    _materialize(candidate, batch)
    assert set(reference.state_dict()) == set(candidate.state_dict())
    reference.load_state_dict(candidate.state_dict(), strict=True)


def _assert_gradient_sets_close(reference, candidate) -> None:
    reference_prediction, reference_parameters, reference_gradients = reference
    candidate_prediction, candidate_parameters, candidate_gradients = candidate
    assert set(reference_parameters) == set(candidate_parameters)
    torch.testing.assert_close(reference_prediction, candidate_prediction, rtol=0.0, atol=0.0)
    for name, left, right in zip(
        (*reference_parameters, "input:centers", "input:module_features", "input:environment_features", "input:queries"),
        reference_gradients,
        candidate_gradients,
        strict=True,
    ):
        if left is None or right is None:
            assert left is right, name
        else:
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0, msg=name)


def test_native_dense_without_policy_uses_no_tree_and_matches_dense_outputs_and_gradients() -> None:
    torch.manual_seed(37)
    config = _config()
    batch = _batch()
    candidate = InterfaceFieldCore(config).eval()
    reference = _dense_reference(config).eval()
    assert isinstance(candidate.backend, AdaptiveCoverPairwiseField)

    def forbidden_tree_build(_encoded):
        pytest.fail("the no-policy Dense path must not build receiver trees")

    candidate.backend.build_case_trees = forbidden_tree_build
    _copy_reference_parameters(reference, candidate, batch)
    reference_gradients = _parameter_and_input_gradients(reference, batch)
    candidate_gradients = _parameter_and_input_gradients(candidate, batch)
    _assert_gradient_sets_close(reference_gradients, candidate_gradients)
    encoded = candidate.encode_case(batch)
    prepared = candidate.prepare(encoded, encoded.module_tokens)
    assert "cover_trees" not in prepared.backend_state
    assert "cover_preparation_ledger" not in prepared.backend_state


def test_native_all_access_policy_is_exact_and_keeps_native_parameter_inventory() -> None:
    torch.manual_seed(43)
    config = _config()
    batch = _batch()
    candidate = InterfaceFieldCore(config).eval()
    reference = _dense_reference(config).eval()
    _copy_reference_parameters(reference, candidate, batch)
    policy = _AllAccessPolicy()
    candidate.set_native_interaction_policy(policy)

    reference_gradients = _parameter_and_input_gradients(reference, batch)
    candidate_gradients = _parameter_and_input_gradients(candidate, batch)
    _assert_gradient_sets_close(reference_gradients, candidate_gradients)

    encoded = candidate.encode_case(batch)
    prepared = candidate.prepare(encoded, encoded.module_tokens)
    assert prepared.backend_state["cover_all_access"] is True
    assert int(prepared.interaction_aux["cover_prepare_dense_fallback"]) == 1
    read = candidate.read(prepared, batch.query_xy, receiver_chunk_size=2)
    assert int(read.interaction_aux["cover_executor_dense_fallback"]) == 4
    assert int(read.interaction_aux["cover_coarse_path_bypass_active"]) == 1
    assert int(read.interaction_aux["cover_local_path_bypass_active"]) == 1
    assert policy.prepared_keys == {"module_states", "environment_states", "global_state"}
    assert policy.encoded_has_target is False
    policy_keys = {key for key in candidate.state_dict() if key.startswith("native_interaction_policy.")}
    assert set(reference.state_dict()) == set(candidate.state_dict()) - policy_keys


def test_partial_native_plan_masks_mm_me_em_before_aggregation_and_records_bypasses() -> None:
    torch.manual_seed(53)
    config = _config()
    batch = _batch()
    reference = _dense_reference(config).eval()
    partial = InterfaceFieldCore(config).eval()
    _copy_reference_parameters(reference, partial, batch)
    partial.set_native_interaction_policy(_PrunedRootPolicy())

    centers = batch.module_centers.detach().clone().requires_grad_(True)
    module_features = batch.module_features.detach().clone().requires_grad_(True)
    environment_features = batch.env_features.detach().clone().requires_grad_(True)
    current = replace(
        batch,
        module_centers=centers,
        module_features=module_features,
        env_features=environment_features,
    )
    encoded = partial.encode_case(current)
    prepared = partial.prepare(encoded, encoded.module_tokens)
    preparation = prepared.interaction_aux
    assert int(preparation["cover_prepare_dense_fallback"]) == 0
    assert int(preparation["cover_prepare_mm_executed_rows"]) < int(preparation["cover_prepare_mm_valid_rows"])
    assert int(preparation["cover_prepare_me_executed_rows"]) < int(preparation["cover_prepare_me_valid_rows"])
    assert int(preparation["cover_prepare_em_executed_rows"]) < int(preparation["cover_prepare_em_valid_rows"])
    assert int(preparation["cover_prepare_coarse_bypass_active"]) == 1
    assert int(preparation["cover_prepare_local_bypass_active"]) == 1

    prediction = partial.decode_queries(
        prepared,
        batch.query_xy,
        batch.query_features,
        receiver_chunk_size=2,
        return_interaction_aux=True,
    )
    prediction_value = prediction["pred_field"]
    ledger = prediction["_interaction_aux"]
    full_chunk_prediction = partial.decode_queries(
        prepared,
        batch.query_xy,
        batch.query_features,
        receiver_chunk_size=7,
    )["pred_field"]
    torch.testing.assert_close(prediction_value, full_chunk_prediction, rtol=2.0e-5, atol=2.0e-6)
    assert int(ledger["cover_executor_dense_fallback"]) == 0
    assert int(ledger["cover_qm_unique_rows"]) == 7 * 2
    assert int(ledger["cover_qe_unique_rows"]) == 7 * 7
    assert int(ledger["cover_coarse_path_bypass_active"]) == 1
    assert int(ledger["cover_local_path_bypass_active"]) == 1
    assert int(ledger["cover_local_bypass_neighbor_rows"]) == int(ledger["local_neighbor_count"].sum())
    assert torch.isfinite(prediction_value).all()
    assert not torch.allclose(prediction_value, _predict(reference, batch))

    typed_context = prepared.backend_state["module_tokens"].sum() + prepared.backend_state["env_tokens"].sum()
    module_gradient, environment_gradient = torch.autograd.grad(
        typed_context, (module_features, environment_features)
    )
    assert torch.isfinite(module_gradient).all() and float(module_gradient.abs().sum()) > 0.0
    assert torch.isfinite(environment_gradient).all() and float(environment_gradient.abs().sum()) > 0.0


def test_partial_native_plan_is_invariant_to_source_permutation_and_quadrature_splitting() -> None:
    torch.manual_seed(67)
    config = _config()
    batch = _batch()
    partial = InterfaceFieldCore(config).eval()
    partial.set_native_interaction_policy(_PrunedRootPolicy())

    def predict(current: BatchData, chunk_size: int) -> torch.Tensor:
        encoded = partial.encode_case(current)
        prepared = partial.prepare(encoded, encoded.module_tokens)
        return partial.decode_queries(
            prepared,
            current.query_xy,
            current.query_features,
            receiver_chunk_size=chunk_size,
        )["pred_field"]

    module_order = torch.tensor([2, 1, 0, 3])
    environment_order = torch.tensor([4, 1, 2, 3, 0, 6, 5, 7])
    permuted = replace(
        batch,
        module_centers=batch.module_centers[:, module_order],
        module_present=batch.module_present[:, module_order],
        module_features=batch.module_features[:, module_order],
        env_coords=batch.env_coords[:, environment_order],
        env_features=batch.env_features[:, environment_order],
        env_weights=batch.env_weights[:, environment_order],
    )
    split_quadrature = replace(
        batch,
        env_coords=torch.cat((batch.env_coords, batch.env_coords[:, -1:]), dim=1),
        env_features=torch.cat((batch.env_features, batch.env_features[:, -1:]), dim=1),
        env_weights=torch.cat(
            (batch.env_weights[:, :-1], batch.env_weights[:, -1:] * 0.5, batch.env_weights[:, -1:] * 0.5),
            dim=1,
        ),
    )

    with torch.no_grad():
        reference = predict(batch, 2)
        tiled = predict(batch, 7)
        permuted_prediction = predict(permuted, 3)
        split_prediction = predict(split_quadrature, 4)

    torch.testing.assert_close(reference, tiled, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(reference, permuted_prediction, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(reference, split_prediction, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(
        batch.env_weights.sum(dim=1), split_quadrature.env_weights.sum(dim=1), rtol=1.0e-6, atol=1.0e-6
    )


def test_native_tree_cache_reuses_geometry_only_and_recomputes_model_state() -> None:
    torch.manual_seed(71)
    core = InterfaceFieldCore(_config()).eval()
    core.set_native_interaction_policy(_AllAccessPolicy())
    original_batch = _batch()

    original_encoded = core.encode_case(original_batch)
    original_prepared = core.prepare(original_encoded, original_encoded.module_tokens)
    original_tree = original_prepared.backend_state["cover_trees"][0]
    assert int(original_prepared.interaction_aux["cover_prepare_tree_cache_misses"]) == 1
    assert int(original_prepared.interaction_aux["cover_prepare_tree_cache_hits"]) == 0

    # Features affect freshly encoded and prepared model states, but not the
    # physical receiver index; that immutable index is the only reused object.
    changed_batch = replace(
        original_batch,
        module_features=original_batch.module_features + 0.75,
        global_context=original_batch.global_context - 0.25,
    )
    changed_encoded = core.encode_case(changed_batch)
    changed_prepared = core.prepare(changed_encoded, changed_encoded.module_tokens)
    assert changed_prepared.backend_state["cover_trees"][0] is original_tree
    assert int(changed_prepared.interaction_aux["cover_prepare_tree_cache_hits"]) == 1
    assert int(changed_prepared.interaction_aux["cover_prepare_tree_cache_misses"]) == 0
    assert not torch.equal(
        original_prepared.backend_state["module_tokens"],
        changed_prepared.backend_state["module_tokens"],
    )
    assert not torch.equal(
        original_prepared.backend_state["env_tokens"],
        changed_prepared.backend_state["env_tokens"],
    )
    assert not original_tree.universe.coordinates.requires_grad
    assert not original_tree.universe.weights.requires_grad
    assert not original_tree.universe.coordinate_scale.requires_grad

    cache_before_direct_dense = core.native_interaction_tree_cache_info()
    core.set_native_interaction_policy(None)
    dense_prepared = core.prepare(changed_encoded, changed_encoded.module_tokens)
    assert "cover_trees" not in dense_prepared.backend_state
    assert core.native_interaction_tree_cache_info() == cache_before_direct_dense


def test_native_tree_cache_invalidates_on_geometry_measure_scale_and_settings() -> None:
    torch.manual_seed(73)
    core = InterfaceFieldCore(_config()).eval()
    backend = core.backend
    encoded = core.encode_case(_batch())
    baseline_tree = backend.build_case_trees(encoded)[0]

    module_centers = encoded.module_centers.clone()
    module_centers[0, 0, 0] += 0.125
    module_present = encoded.module_present.clone()
    module_present[0, 3] = 1.0
    environment_coords = encoded.env_coords.clone()
    environment_coords[0, 0, 1] += 0.125
    environment_weights = encoded.env_weights.clone()
    environment_weights[0, 0] += 0.125
    coordinate_scale = encoded.coordinate_scale.clone()
    coordinate_scale[..., 0] += 1.0

    perturbed_designs = (
        replace(encoded, module_centers=module_centers),
        replace(encoded, module_present=module_present),
        replace(encoded, env_coords=environment_coords),
        replace(encoded, env_weights=environment_weights),
        replace(encoded, coordinate_scale=coordinate_scale),
    )
    for changed in perturbed_designs:
        assert backend.build_case_trees(changed)[0] is not baseline_tree

    anchor_coords = encoded.env_coords[:, :6].clone()
    anchor_weights = encoded.env_weights[:, :6].clone()
    anchor_roles = torch.arange(6, dtype=torch.long).reshape(1, 6)
    anchored = replace(
        encoded,
        receiver_anchor_coords=anchor_coords,
        receiver_anchor_weights=anchor_weights,
        receiver_anchor_roles=anchor_roles,
    )
    anchor_tree = backend.build_case_trees(anchored)[0]
    assert anchor_tree is not baseline_tree
    for changed in (
        replace(anchored, receiver_anchor_coords=anchor_coords + 0.125),
        replace(anchored, receiver_anchor_weights=anchor_weights + 0.125),
        replace(anchored, receiver_anchor_roles=anchor_roles + 1),
    ):
        assert backend.build_case_trees(changed)[0] is not anchor_tree

    double_encoded = replace(
        encoded,
        module_centers=encoded.module_centers.double(),
        module_present=encoded.module_present.double(),
        env_coords=encoded.env_coords.double(),
        env_weights=encoded.env_weights.double(),
        coordinate_scale=encoded.coordinate_scale.double(),
    )
    assert backend.build_case_trees(double_encoded)[0] is not baseline_tree

    original_capacity = backend.max_nodes
    try:
        backend.max_nodes = original_capacity - 1
        assert backend.build_case_trees(encoded)[0] is not baseline_tree
    finally:
        backend.max_nodes = original_capacity


def test_native_tree_cache_does_not_reuse_autograd_geometry_graphs() -> None:
    torch.manual_seed(79)
    core = InterfaceFieldCore(_config()).eval()
    batch = _batch()
    centers = batch.module_centers.detach().clone().requires_grad_(True)
    encoded = core.encode_case(replace(batch, module_centers=centers))

    first_tree = core.backend.build_case_trees(encoded)[0]
    second_tree = core.backend.build_case_trees(encoded)[0]
    assert first_tree is not second_tree
    assert first_tree.universe.coordinates.requires_grad
    assert second_tree.universe.coordinates.requires_grad
    assert core.backend.case_tree_cache_info()["hits"] == 0

    root = first_tree.nodes[0]
    assert root.left is not None and root.right is not None and root.split_axis is not None
    left = first_tree.nodes[root.left].anchor_indices
    right = first_tree.nodes[root.right].anchor_indices
    boundary = (
        first_tree.universe.coordinates[list(left), root.split_axis].mean()
        + first_tree.universe.coordinates[list(right), root.split_axis].mean()
    ) / 2.0
    query = centers.new_zeros((1, 3))
    query[0, root.split_axis] = boundary.detach()
    plan = AdaptiveCoverPlan.full_access(first_tree, encoded.module_present[0], int(encoded.env_coords.shape[1]))
    split_gates = plan.split_gates.clone()
    split_gates[0] = 1.0
    plan = replace(plan, split_gates=split_gates)
    route_weight = plan.access(query).receiver_group[0, root.left]
    gradient = torch.autograd.grad(route_weight, centers)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0
