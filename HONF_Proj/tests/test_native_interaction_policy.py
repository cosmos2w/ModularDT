"""Native Dense policy opt-in, exact all-access, and typed transport coverage."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from honf_forward_core.config import BatchData, InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.interface_fields.adaptive_cover_field import AdaptiveCoverPairwiseField
from honf_forward_core.interface_fields.adaptive_interaction_cover import (
    AdaptiveCoverPlan,
    CaseLocalReceiverTree,
    InteractionContext,
    MechanismPlan,
    ReceiverAnchorUniverse,
)
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.interface_fields.dense_pairwise import DensePairwiseField
from honf_forward_core.interface_fields.input_cover_organizer import InputOnlyCoverOrganizer


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


def test_fixed_typed_plan_has_dense_parity_and_phase_mask_changes_output() -> None:
    torch.manual_seed(47)
    config = _config()
    batch = _batch()
    candidate = InterfaceFieldCore(config).eval()
    reference = _dense_reference(config).eval()
    _copy_reference_parameters(reference, candidate, batch)

    encoded = candidate.encode_case(batch)
    trees = candidate.backend.build_case_trees(encoded)
    plan = MechanismPlan.full_access(
        trees[0], encoded.module_present[0], int(encoded.env_coords.shape[1]), phase="P0"
    )
    plans = (plan,)
    context = InteractionContext(phase="P0", receiver_role="query")

    def fixed_prediction(core: InterfaceFieldCore, current: BatchData, selected_plans=plans) -> torch.Tensor:
        current_encoded = core.encode_case(current)
        prepared = core.prepare(
            current_encoded,
            current_encoded.module_tokens,
            fixed_cover_plans=selected_plans,
            interaction_context=context,
        )
        assert len(prepared.backend_state["cover_plans"]) == 1
        return core.decode_queries(
            prepared,
            current.query_xy,
            current.query_features,
            receiver_chunk_size=2,
            interaction_context=context,
        )["pred_field"]

    with torch.no_grad():
        expected = _predict(reference, batch)
        actual = fixed_prediction(candidate, batch)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    # Force the cover reader for a recursive identity split: the two active
    # children have the same source permissions, so their access weights must
    # sum back to the parent without changing either values or query gradients.
    candidate.backend.set_cover_mode("external")
    identity_plan = plan.with_split(0, 1.0)
    with torch.no_grad():
        identity_prediction = fixed_prediction(candidate, batch, (identity_plan,))
    torch.testing.assert_close(identity_prediction, expected, rtol=2.0e-5, atol=2.0e-6)
    identity_queries = batch.query_xy.detach().clone().requires_grad_(True)
    reference_queries = batch.query_xy.detach().clone().requires_grad_(True)
    identity_current = replace(batch, query_xy=identity_queries)
    reference_current = replace(batch, query_xy=reference_queries)
    identity_gradient_prediction = fixed_prediction(candidate, identity_current, (identity_plan,))
    reference_gradient_prediction = _predict(reference, reference_current)
    identity_query_gradient = torch.autograd.grad(
        identity_gradient_prediction.square().sum(), identity_queries
    )[0]
    reference_query_gradient = torch.autograd.grad(
        reference_gradient_prediction.square().sum(), reference_queries
    )[0]
    torch.testing.assert_close(
        identity_query_gradient, reference_query_gradient, rtol=2.0e-5, atol=2.0e-6
    )

    closed_query_plan = plan.with_permission(
        "QM", torch.zeros_like(plan.permission_matrix("QM")), phase="P0"
    )
    with torch.no_grad():
        masked = fixed_prediction(candidate, batch, (closed_query_plan,))
    assert not torch.allclose(masked, actual)

    closed_environment_plan = plan.with_permission(
        "QE", torch.zeros_like(plan.permission_matrix("QE")), phase="P0"
    )
    closed_encoded = candidate.encode_case(batch)
    closed_prepared = candidate.prepare(
        closed_encoded,
        closed_encoded.module_tokens,
        fixed_cover_plans=(closed_environment_plan,),
        interaction_context=context,
    )
    closed_read = candidate.read(
        closed_prepared,
        batch.query_xy,
        receiver_chunk_size=7,
        interaction_context=context,
    )
    assert torch.isfinite(closed_read.context).all()
    assert int(closed_read.interaction_aux["cover_qe_unique_source_receiver_pairs"]) == 0
    assert int(closed_read.interaction_aux["cover_environment_fallback_queries"]) == 0
    candidate.backend.set_cover_mode("full_access")

    # Ordinary decode calls do not request route maps or interaction aux. The
    # full-access Dense path should keep the measured row ledger but skip the
    # secondary cover-access/frontier diagnostic computation.
    fast_encoded = candidate.encode_case(batch)
    fast_prepared = candidate.prepare(
        fast_encoded,
        fast_encoded.module_tokens,
        fixed_cover_plans=plans,
        interaction_context=context,
    )
    original_query_accesses = candidate.backend._query_accesses
    original_work_aux = candidate.backend._cover_query_work_aux

    def diagnostics_are_disabled(*_args, **_kwargs):
        raise AssertionError("diagnostics-off Dense decode recomputed cover diagnostics")

    candidate.backend._query_accesses = diagnostics_are_disabled
    candidate.backend._cover_query_work_aux = diagnostics_are_disabled
    try:
        with torch.no_grad():
            fast_output = candidate.decode_queries(
                fast_prepared,
                batch.query_xy,
                batch.query_features,
                interaction_context=context,
            )["pred_field"]
    finally:
        candidate.backend._query_accesses = original_query_accesses
        candidate.backend._cover_query_work_aux = original_work_aux
    torch.testing.assert_close(fast_output, expected, rtol=0.0, atol=0.0)

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
    fixed_value = fixed_prediction(candidate, current)
    parameters = dict(candidate.named_parameters())
    fixed_gradients = torch.autograd.grad(
        fixed_value.square().sum(),
        (*parameters.values(), centers, module_features, environment_features, queries),
        allow_unused=True,
    )
    fixed_result = fixed_value, parameters, fixed_gradients
    _assert_gradient_sets_close(
        _parameter_and_input_gradients(reference, batch), fixed_result
    )


def test_fixed_typed_native_plan_preserves_action_when_quadrature_atom_is_split() -> None:
    torch.manual_seed(49)
    core = InterfaceFieldCore(_config()).eval()
    base = _batch()
    # Represent one source quadrature atom by two identical half-weight
    # source slots. Native receiver anchors are generated from that same
    # source catalogue, as they are in the default core contract.
    split = replace(
        base,
        env_coords=torch.cat((base.env_coords[:, :-1], base.env_coords[:, -1:], base.env_coords[:, -1:]), dim=1),
        env_features=torch.cat((base.env_features[:, :-1], base.env_features[:, -1:], base.env_features[:, -1:]), dim=1),
        env_weights=torch.cat(
            (base.env_weights[:, :-1], base.env_weights[:, -1:] * 0.5, base.env_weights[:, -1:] * 0.5),
            dim=1,
        ),
    )
    context = InteractionContext(phase="P0", receiver_role="query")

    def plan_for(current: BatchData) -> MechanismPlan:
        encoded = core.encode_case(current)
        tree = core.backend.build_case_trees(encoded)[0]
        plan = MechanismPlan.full_access(
            tree, encoded.module_present[0], int(encoded.env_coords.shape[1]), phase="P0"
        )
        permissions = {}
        for key, value in plan.permissions.items():
            if key.mechanism not in {"ME", "QE"}:
                permissions[key] = value
                continue
            masked = value.clone()
            # Keep the split atom open in both representations and close one
            # unrelated source so the full native path executes a real mask.
            masked[:, 6] = 0.0
            permissions[key] = masked
        return MechanismPlan(
            tree,
            plan.split_gates,
            plan.module_present,
            int(encoded.env_coords.shape[1]),
            permissions,
        )

    def predict(current: BatchData, plan: MechanismPlan) -> torch.Tensor:
        encoded = core.encode_case(current)
        prepared = core.prepare(
            encoded,
            encoded.module_tokens,
            fixed_cover_plans=(plan,),
            interaction_context=context,
        )
        return core.decode_queries(
            prepared,
            current.query_xy,
            current.query_features,
            receiver_chunk_size=2,
            interaction_context=context,
        )["pred_field"]

    original_plan = plan_for(base)
    split_plan = plan_for(split)
    assert original_plan.permission_matrix("QE", phase="P0")[:, -1].eq(1).all()
    assert split_plan.permission_matrix("QE", phase="P0")[:, -2:].eq(1).all()
    assert original_plan.permission_matrix("QE", phase="P0")[:, 6].eq(0).all()
    assert split_plan.permission_matrix("QE", phase="P0")[:, 6].eq(0).all()
    with torch.no_grad():
        original = predict(base, original_plan)
        duplicated = predict(split, split_plan)
    torch.testing.assert_close(duplicated, original, rtol=2.0e-5, atol=2.0e-6)


def test_differentiable_hard_full_access_uses_masked_path_without_changing_native_grads() -> None:
    torch.manual_seed(53)
    config = _config()
    batch = _batch()
    candidate = InterfaceFieldCore(config).eval()
    reference = _dense_reference(config).eval()
    _copy_reference_parameters(reference, candidate, batch)

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
    encoded = candidate.encode_case(current)
    tree = candidate.backend.build_case_trees(encoded)[0]
    phase = "P0"
    plan = MechanismPlan.full_access(
        tree, encoded.module_present[0], int(encoded.env_coords.shape[1]), phase=phase
    )
    permission_leaves: dict[str, torch.Tensor] = {}
    for mechanism in ("MM", "ME", "EM", "QM", "QE"):
        leaf = plan.permission_matrix(mechanism, phase=phase).detach().clone().requires_grad_(True)
        permission_leaves[mechanism] = leaf
        plan = plan.with_permission(mechanism, leaf, phase=phase)

    # This also guards the historical read bypass, which ran even when the
    # backend's explicit mode was full_access.
    candidate.backend.set_cover_mode("full_access")
    context = InteractionContext(phase=phase, receiver_role="query")
    prepared = candidate.prepare(
        encoded,
        encoded.module_tokens,
        fixed_cover_plans=(plan,),
        interaction_context=context,
    )
    assert prepared.backend_state["cover_all_access"] is True
    preparation_ledger = prepared.backend_state["cover_preparation_ledger"]
    assert preparation_ledger["cover_prepare_gradient_bridge"] == 1
    assert preparation_ledger["cover_prepare_executor_native_dense"] == 1
    assert preparation_ledger["cover_prepare_executor_dense_masked"] == 1
    for mechanism in ("mm", "me", "em"):
        prefix = f"cover_prepare_{mechanism}_"
        native_rows = preparation_ledger[f"{prefix}gradient_bridge_native_rows"]
        surrogate_rows = preparation_ledger[f"{prefix}gradient_bridge_surrogate_rows"]
        assert native_rows > 0 and surrogate_rows > 0
        assert preparation_ledger[f"{prefix}actual_rows"] == native_rows + surrogate_rows
    read = candidate.read(
        prepared, queries, receiver_chunk_size=2, interaction_context=context
    )
    assert int(read.interaction_aux["cover_qm_executor_dense_masked"]) == 1
    assert int(read.interaction_aux["cover_qe_executor_dense_masked"]) == 1
    assert int(read.interaction_aux["cover_gradient_bridge_surrogate"]) == 1
    assert int(read.interaction_aux["cover_executor_native_dense"]) == 1
    assert int(read.interaction_aux["cover_qm_actual_rows"]) == (
        int(read.interaction_aux["cover_qm_gradient_bridge_native_rows"])
        + int(read.interaction_aux["cover_qm_gradient_bridge_surrogate_rows"])
    )
    prediction = candidate.decode_queries(
        prepared,
        queries,
        current.query_features,
        receiver_chunk_size=2,
        interaction_context=context,
    )["pred_field"]
    reference_prediction = _predict(reference, current)
    torch.testing.assert_close(prediction, reference_prediction, rtol=0.0, atol=0.0)

    candidate_parameters = dict(candidate.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    assert candidate_parameters.keys() == reference_parameters.keys()
    candidate_gradients = torch.autograd.grad(
        prediction.square().sum(),
        (*candidate_parameters.values(), centers, module_features, environment_features, queries,
         *permission_leaves.values()),
        allow_unused=True,
    )
    reference_gradients = torch.autograd.grad(
        reference_prediction.square().sum(),
        (*reference_parameters.values(), centers, module_features, environment_features, queries),
        allow_unused=True,
    )
    for name, actual, expected in zip(
        (*candidate_parameters, "input:centers", "input:module_features", "input:environment_features", "input:queries"),
        candidate_gradients[: len(candidate_parameters) + 4],
        reference_gradients,
        strict=True,
    ):
        assert actual is not None and expected is not None, name
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, msg=name)
    permission_gradients = candidate_gradients[len(candidate_parameters) + 4 :]
    for mechanism, gradient in zip(permission_leaves, permission_gradients, strict=True):
        assert gradient is not None, mechanism
        assert bool(torch.isfinite(gradient).all()), mechanism
        assert float(gradient.abs().sum()) > 0.0, mechanism


def test_hard_endpoint_split_surrogate_receives_native_prediction_gradient() -> None:
    torch.manual_seed(59)
    config = _config()
    batch = _batch()
    candidate = InterfaceFieldCore(config).eval()
    reference = _dense_reference(config).eval()
    _copy_reference_parameters(reference, candidate, batch)
    candidate.backend.set_cover_mode("external")
    candidate.backend.set_cover_executor("dense_masked")

    encoded = candidate.encode_case(batch)
    tree = candidate.backend.build_case_trees(encoded)[0]
    root = tree.nodes[0]
    assert root.left is not None
    phase = "P0"
    hard_gate = torch.zeros((), requires_grad=True)
    plan = MechanismPlan.full_access(
        tree, encoded.module_present[0], int(encoded.env_coords.shape[1]), phase=phase
    ).with_split(0, hard_gate)
    child_qm = plan.permission_matrix("QM", phase=phase).clone()
    # At the all-access root the hard forward retains every source. Opening
    # the left child would omit one source, so its protected-output derivative
    # tests the endpoint straight-through gate on the actual reader.
    child_qm[root.left, 0] = 0.0
    plan = plan.with_permission("QM", child_qm, phase=phase)
    context = InteractionContext(phase=phase, receiver_role="query")
    prepared = candidate.prepare(
        encoded,
        encoded.module_tokens,
        fixed_cover_plans=(plan,),
        interaction_context=context,
    )
    prediction = candidate.decode_queries(
        prepared,
        batch.query_xy,
        batch.query_features,
        receiver_chunk_size=2,
        interaction_context=context,
    )["pred_field"]
    expected = _predict(reference, batch)
    torch.testing.assert_close(prediction, expected, rtol=2.0e-5, atol=2.0e-6)
    gate_gradient = torch.autograd.grad(prediction.square().sum(), hard_gate)[0]
    assert torch.isfinite(gate_gradient)
    assert float(gate_gradient.abs()) > 0.0

def test_fixed_cover_rejects_same_count_environment_source_permutation() -> None:
    candidate = InterfaceFieldCore(_config()).eval()
    batch = _batch()
    encoded = candidate.encode_case(batch)
    tree = candidate.backend.build_case_trees(encoded)[0]
    plan = MechanismPlan.full_access(
        tree, encoded.module_present[0], int(encoded.env_coords.shape[1]), phase="P0"
    )
    order = torch.tensor([4, 1, 2, 3, 0, 6, 5, 7])
    permuted = replace(batch, env_coords=batch.env_coords[:, order])
    permuted_encoded = candidate.encode_case(permuted)

    with pytest.raises(ValueError, match="environmental source coordinate order"):
        candidate.prepare(
            permuted_encoded,
            permuted_encoded.module_tokens,
            fixed_cover_plans=(plan,),
            interaction_context=InteractionContext(phase="P0", receiver_role="query"),
        )


def test_fixed_cover_read_context_must_match_prepared_phase_but_may_change_role() -> None:
    core = InterfaceFieldCore(_config()).eval()
    batch = _batch()
    encoded = core.encode_case(batch)
    tree = core.backend.build_case_trees(encoded)[0]
    plan = MechanismPlan.full_access(
        tree, encoded.module_present[0], int(encoded.env_coords.shape[1]), phase="P0"
    )
    prepared = core.prepare(
        encoded,
        encoded.module_tokens,
        fixed_cover_plans=(plan,),
        interaction_context=InteractionContext(phase="P0", receiver_role="p0_port"),
    )
    assert prepared.backend_state["cover_fixed_plans_explicit"] is True

    # Multiple receiver roles may consume one P0 preparation.
    same_phase_read = core.read(
        prepared,
        batch.query_xy,
        interaction_context=InteractionContext(phase="P0", receiver_role="p0_outside"),
    )
    assert torch.isfinite(same_phase_read.context).all()

    wrong_phase = InteractionContext(phase="P1", receiver_role="p1_refinement")
    with pytest.raises(ValueError, match="fixed cover plans cannot be read in a different interaction phase"):
        core.read(prepared, batch.query_xy, interaction_context=wrong_phase)
    with pytest.raises(ValueError, match="fixed cover plans cannot be read in a different interaction phase"):
        core.decode_queries(
            prepared,
            batch.query_xy,
            batch.query_features,
            interaction_context=wrong_phase,
        )


def test_cover_raw_path_counter_uses_exact_int64_node_products() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    universe = ReceiverAnchorUniverse(
        coordinates=torch.tensor(
            [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            device=device,
        ),
        weights=torch.ones(4, device=device),
        roles=torch.zeros(4, dtype=torch.long, device=device),
        coordinate_scale=torch.ones(3, device=device),
    )
    tree = CaseLocalReceiverTree.build(
        universe, max_nodes=3, min_leaf_anchors=1, overlap_fraction=0.05
    )
    assert tree.nodes[0].left is not None
    module_present = torch.tensor([1.0, 1.0, 0.0], device=device)
    plan = MechanismPlan.full_access(tree, module_present, 512, phase="P0")
    query_count = 40_960
    receivers = torch.zeros((1, query_count, 3), device=device)
    # These tensors are used only for shape accounting in this helper. Expand
    # singleton storage to keep the large-Q exact-count check inexpensive.
    module_access = torch.zeros((1, 1, 3), device=device).expand(1, query_count, 3)
    environment_prior = torch.zeros((1, 1, 512), device=device).expand(1, query_count, 512)
    encoded = type("EncodedShape", (), {"module_present": module_present[None]})()
    core = InterfaceFieldCore(_config()).to(device)

    aux = core.backend._cover_query_work_aux(
        (plan,),
        encoded,
        receivers,
        module_access,
        environment_prior,
        module_actual_rows=query_count * 3,
        environment_actual_rows=query_count * 512,
        module_unique_rows=query_count * 2,
        environment_unique_rows=query_count * 512,
        phase="P0",
        executor="dense_masked",
    )

    assert int(aux["cover_qm_raw_paths"]) == query_count * 2
    # This exceeds 2**24, so a float32 GEMM count can no longer be trusted.
    assert int(aux["cover_qe_raw_paths"]) == query_count * 512


def test_dense_masked_and_rectangular_subset_report_evaluated_and_unique_rows() -> None:
    torch.manual_seed(49)
    core = InterfaceFieldCore(_config()).eval()
    batch = _batch()
    encoded = core.encode_case(batch)
    tree = core.backend.build_case_trees(encoded)[0]
    plan = MechanismPlan.full_access(
        tree, encoded.module_present[0], int(encoded.env_coords.shape[1]), phase="P0"
    )
    qm = plan.permission_matrix("QM", phase="P0")
    qe = plan.permission_matrix("QE", phase="P0")
    qm[:, 1:] = 0.0
    qe[:, 2:] = 0.0
    plan = plan.with_permission("QM", qm, phase="P0").with_permission("QE", qe, phase="P0")
    context = InteractionContext(phase="P0", receiver_role="query")

    core.backend.set_cover_mode("external")
    core.backend.set_cover_executor("dense_masked")
    dense_prepared = core.prepare(
        encoded, encoded.module_tokens, fixed_cover_plans=(plan,), interaction_context=context
    )
    dense_read = core.read(
        dense_prepared,
        batch.query_xy,
        receiver_chunk_size=7,
        interaction_context=context,
    )

    core.backend.set_cover_executor("rectangular_subset")
    subset_prepared = core.prepare(
        encoded, encoded.module_tokens, fixed_cover_plans=(plan,), interaction_context=context
    )
    subset_read = core.read(
        subset_prepared,
        batch.query_xy,
        receiver_chunk_size=7,
        interaction_context=context,
    )

    torch.testing.assert_close(dense_read.context, subset_read.context, rtol=2.0e-5, atol=2.0e-6)
    dense_aux = dense_read.interaction_aux
    subset_aux = subset_read.interaction_aux
    assert int(dense_aux["cover_qm_executor_dense_masked"]) == 1
    assert int(dense_aux["cover_qe_executor_dense_masked"]) == 1
    assert int(subset_aux["cover_qm_executor_rectangular_subset"]) == 1
    assert int(subset_aux["cover_qe_executor_rectangular_subset"]) == 1
    assert int(dense_aux["cover_qm_actual_rows"]) == 7 * 4
    assert int(dense_aux["cover_qm_unique_source_receiver_pairs"]) == 7
    assert int(dense_aux["cover_qm_padded_rows"]) == 7 * 3
    assert int(dense_aux["cover_qe_actual_rows"]) == 7 * 8
    assert int(dense_aux["cover_qe_unique_source_receiver_pairs"]) == 7 * 2
    assert int(dense_aux["cover_qe_padded_rows"]) == 7 * 6
    assert int(subset_aux["cover_qm_actual_rows"]) == int(subset_aux["cover_qm_unique_rows"]) == 7
    assert int(subset_aux["cover_qe_actual_rows"]) == int(subset_aux["cover_qe_unique_rows"]) == 7 * 2
    assert int(dense_aux["cover_qm_source_union_count"]) == 1
    assert int(dense_aux["cover_qe_source_union_count"]) == 2
    assert int(dense_aux["cover_qe_nonredundant_packets"]) == 1
    assert int(dense_aux["cover_environment_fallback_queries"]) == 0


def test_partial_native_plan_masks_mm_me_em_before_aggregation_and_records_bypasses() -> None:
    torch.manual_seed(53)
    config = _config()
    batch = _batch()
    reference = _dense_reference(config).eval()
    partial = InterfaceFieldCore(config).eval()
    _copy_reference_parameters(reference, partial, batch)
    partial.set_native_interaction_policy(_PrunedRootPolicy())
    partial.backend.set_cover_executor("rectangular_subset")

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
    assert int(preparation["cover_prepare_executor_rectangular_subset"]) == 1
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


def test_fixed_plan_rebind_keeps_live_anchor_geometry_gradients() -> None:
    torch.manual_seed(83)
    core = InterfaceFieldCore(_config()).eval()
    batch = _batch()
    baseline_encoded = core.encode_case(batch)
    baseline_tree = core.backend.build_case_trees(baseline_encoded)[0]
    root = baseline_tree.nodes[0]
    assert root.left is not None and root.right is not None and root.split_axis is not None
    frozen = MechanismPlan.full_access(
        baseline_tree,
        baseline_encoded.module_present[0],
        int(baseline_encoded.env_coords.shape[1]),
    ).with_split(0, 1.0)
    query_module = torch.zeros_like(frozen.permission_matrix("QM"))
    query_module[root.left, 0] = 1.0
    frozen = frozen.with_permission("QM", query_module, phase="P0")

    centers = batch.module_centers.detach().clone().requires_grad_(True)
    encoded = core.encode_case(replace(batch, module_centers=centers))
    rebound = core.backend.rebind_fixed_plans(encoded, (frozen,))[0]
    assert rebound.tree.nodes == frozen.tree.nodes
    assert rebound.tree.universe.coordinates.requires_grad
    live_root = rebound.tree.nodes[0]
    assert live_root.left is not None and live_root.right is not None and live_root.split_axis is not None
    left_ids = rebound.tree.nodes[live_root.left].anchor_indices
    right_ids = rebound.tree.nodes[live_root.right].anchor_indices
    live_coordinates = rebound.tree.universe.coordinates
    boundary = (
        live_coordinates[list(left_ids), live_root.split_axis].mean()
        + live_coordinates[list(right_ids), live_root.split_axis].mean()
    ) / 2.0
    query = centers.new_zeros((1, 3))
    query[0, live_root.split_axis] = boundary.detach()
    query.requires_grad_(True)
    routed_permission = rebound.access_for("QM", query, phase="P0")
    center_gradient, query_gradient = torch.autograd.grad(
        routed_permission.sum(), (centers, query)
    )

    assert torch.isfinite(center_gradient).all() and float(center_gradient.abs().sum()) > 0.0
    assert torch.isfinite(query_gradient).all() and float(query_gradient.abs().sum()) > 0.0


def test_dense_masked_partial_qe_support_has_finite_live_position_gradients() -> None:
    torch.manual_seed(89)
    core = InterfaceFieldCore(_config()).eval()
    core.backend.set_cover_mode("external")
    core.backend.set_cover_executor("dense_masked")
    batch = _batch()
    centers = batch.module_centers.detach().clone().requires_grad_(True)
    queries = batch.query_xy.detach().clone().requires_grad_(True)
    current = replace(batch, module_centers=centers, query_xy=queries)
    encoded = core.encode_case(current)
    tree = core.backend.build_case_trees(encoded)[0]
    context = InteractionContext(phase="P2", receiver_role="p2_field")
    plan = MechanismPlan.full_access(
        tree, encoded.module_present[0], int(encoded.env_coords.shape[1]), phase="P2"
    ).with_split(0, 1.0)
    root = tree.nodes[0]
    assert root.left is not None
    permission = plan.permission_matrix("QE", phase="P2").clone()
    permission[root.left, -3:] = 0.0
    plan = plan.with_permission("QE", permission, phase="P2")

    prepared = core.prepare(
        encoded,
        encoded.module_tokens,
        fixed_cover_plans=(plan,),
        interaction_context=context,
    )
    read = core.read(
        prepared,
        queries,
        receiver_chunk_size=2,
        interaction_context=context,
    )
    position_gradients = torch.autograd.grad(
        read.context.square().sum(), (centers, queries), allow_unused=True
    )

    for gradient in position_gradients:
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.abs().sum()) > 0.0


def test_straight_through_organizer_gets_gradient_through_typed_native_execution() -> None:
    torch.manual_seed(97)
    core = InterfaceFieldCore(_config()).eval()
    core.backend.set_cover_mode("external")
    core.backend.set_cover_executor("dense_masked")
    batch = _batch()
    encoded = core.encode_case(batch)
    trees = core.backend.build_case_trees(encoded)
    organizer = InputOnlyCoverOrganizer(
        state_dim=int(encoded.module_tokens.shape[-1]),
        module_feature_dim=int(encoded.module_features.shape[-1]),
        environment_feature_dim=int(encoded.env_features.shape[-1]),
        hidden_dim=12,
    ).train()
    # Ensure the forward has genuine omissions, independent of random logits;
    # their surrogate derivatives still come from the live sigmoid scores.
    organizer.source_threshold = 0.999999
    context = InteractionContext(phase="P2", receiver_role="p2_field")
    prepared_state = {
        "module_states": encoded.module_tokens,
        "environment_states": encoded.env_tokens,
        "global_state": encoded.global_token,
    }
    soft_plans = organizer.plan_cases(encoded, prepared_state, trees)
    assert any(
        bool(((plan.split_gates > 0) & (plan.split_gates < 1)).any())
        or any(
            bool(((plan.permission_matrix(mechanism) > 0) & (plan.permission_matrix(mechanism) < 1)).any())
            for mechanism in ("MM", "ME", "EM", "QM", "QE")
        )
        for plan in soft_plans
    )
    plans = organizer.plan_cases(
        encoded, prepared_state, trees, straight_through_hard=True
    )
    assert len(plans) == 1
    for plan in plans:
        assert set(plan.split_gates.detach().unique().tolist()) <= {0.0, 1.0}
        for mechanism in ("MM", "ME", "EM", "QM", "QE"):
            assert set(plan.permission_matrix(mechanism).detach().unique().tolist()) <= {0.0, 1.0}
    assert any(
        bool((plan.permission_matrix(mechanism) == 0).any())
        for plan in plans for mechanism in ("MM", "ME", "EM", "QM", "QE")
    )

    prepared = core.prepare(
        encoded,
        encoded.module_tokens,
        fixed_cover_plans=plans,
        interaction_context=context,
    )
    prediction = core.decode_queries(
        prepared,
        batch.query_xy,
        batch.query_features,
        receiver_chunk_size=2,
        interaction_context=context,
    )["pred_field"]
    named_parameters = tuple(organizer.named_parameters())
    gradients = torch.autograd.grad(
        prediction.square().sum(),
        tuple(parameter for _, parameter in named_parameters),
        allow_unused=True,
    )
    active_gradients = [gradient for gradient in gradients if gradient is not None]
    assert active_gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in active_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in active_gradients) > 0.0
    assert any(
        name.startswith("pair_scorers.")
        and gradient is not None
        and float(gradient.abs().sum()) > 0.0
        for (name, _), gradient in zip(named_parameters, gradients, strict=True)
    )
