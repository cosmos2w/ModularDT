from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from honf_forward_core.content_hypergraph.contracts import QueryBatch, SceneBatch
from honf_forward_core.inherited_attention import (
    HyperedgeReadoutCoreV1,
    HyperedgeReceiverReadout,
    InheritedQueryBatch,
    MomentHyperedgeCoreV1,
    PreparedHyperedgeReadoutValues,
    ReceiverCover,
    signed_geometry_features,
    smoothstep_grid_receiver_cover,
    with_receiver_cover,
)


def _prepared(values: torch.Tensor, anchor_ids: torch.Tensor | None = None):
    batch, anchors = values.shape[:2]
    if anchor_ids is None:
        anchor_ids = torch.arange(anchors, dtype=torch.long)[None].expand(batch, -1).clone() + 31
    return PreparedHyperedgeReadoutValues(
        values=values,
        anchor_ids=anchor_ids,
        source_state=values.new_zeros(batch, anchors, 128),
        frame_key="readout-frame",
        scene_version="scene-v1",
        owner_token="owner-v1",
    )


def _cover(
    indices: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    validity: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
):
    batch, queries, slots = indices.shape
    if validity is None:
        validity = torch.ones(batch, queries, slots, dtype=torch.bool, device=indices.device)
    return ReceiverCover(
        anchor_indices=indices.to(torch.long),
        anchor_ids=ids.to(torch.long),
        weights=weights,
        validity=validity,
        query_ids=torch.arange(700, 700 + queries, device=indices.device)[None].expand(batch, -1).clone(),
        frame_key="readout-frame",
        query_mask=query_mask,
    )


def _reader(reduction: str = "edge_first", *, geometry_mode: str = "native_signed", access: str = "local_cover"):
    return HyperedgeReceiverReadout(
        coordinate_dim=2,
        reduction=reduction,
        access=access,
        geometry_mode=geometry_mode,
        init_seed=73,
    )


def _local(
    reader,
    prepared,
    query_states,
    query_coordinates,
    anchor_coordinates,
    cover,
    *,
    anchor_mask=None,
    query_mask=None,
    collect_exact_work_counts=False,
    activation=F.silu,
    return_trace=False,
):
    return reader.local_correction(
        prepared=prepared,
        query_states=query_states,
        query_coordinates=query_coordinates,
        anchor_coordinates=anchor_coordinates,
        anchor_mask=anchor_mask,
        frame_scale=query_states.new_ones(query_states.shape[0], 2),
        cover=cover,
        query_mask=query_mask,
        collect_exact_work_counts=collect_exact_work_counts,
        activation=activation,
        return_trace=return_trace,
    )


def test_signed_geometry_features_have_fixed_signed_channel_order():
    relative = torch.tensor([[[0.0, -0.5]]], dtype=torch.float64)
    actual = signed_geometry_features(relative)
    x = relative[..., 1]
    expected = torch.stack(
        (
            torch.asinh(relative[..., 0]),
            torch.sin(torch.pi * relative[..., 0]),
            torch.cos(torch.pi * relative[..., 0]),
            torch.sin(torch.pi * relative[..., 0] / 2),
            torch.cos(torch.pi * relative[..., 0] / 2),
            torch.sin(torch.pi * relative[..., 0] / 4),
            torch.cos(torch.pi * relative[..., 0] / 4),
            torch.asinh(x),
            torch.sin(torch.pi * x),
            torch.cos(torch.pi * x),
            torch.sin(torch.pi * x / 2),
            torch.cos(torch.pi * x / 2),
            torch.sin(torch.pi * x / 4),
            torch.cos(torch.pi * x / 4),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0, 8] < 0  # signed sine channel is not made absolute


def test_pool_first_and_edge_first_agree_for_affine_activation_and_subprobability_weights():
    pool = _reader("pool_first").eval()
    edge = _reader("edge_first").eval()
    edge.load_state_dict(pool.state_dict())
    torch.manual_seed(91)
    with torch.no_grad():
        pool.output_projection.weight.normal_(mean=0.0, std=0.03)
        edge.load_state_dict(pool.state_dict())
    prepared = _prepared(torch.randn(1, 2, 64))
    query_states = torch.randn(1, 2, 128)
    query_coordinates = torch.tensor([[[0.12, -0.21], [0.7, 0.4]]])
    anchor_coordinates = torch.tensor([[[0.0, 0.0], [1.0, 0.8]]])
    query_mask = torch.tensor([[True, False]])
    cover = _cover(
        torch.tensor([[[0, 0, 1], [1, 1, 1]]]),
        torch.tensor([[[31, 31, 32], [32, 32, 32]]]),
        torch.tensor([[[0.20, 0.10, 0.35], [0.0, 0.0, 0.0]]]),
        validity=torch.tensor([[[True, True, True], [False, False, False]]]),
        query_mask=query_mask,
    )
    pool_result, _, pool_work = _local(
        pool, prepared, query_states, query_coordinates, anchor_coordinates, cover,
        query_mask=query_mask, collect_exact_work_counts=True, activation=lambda x: x,
    )
    edge_result, _, edge_work = _local(
        edge, prepared, query_states, query_coordinates, anchor_coordinates, cover,
        query_mask=query_mask, collect_exact_work_counts=True, activation=lambda x: x,
    )
    torch.testing.assert_close(pool_result, edge_result, rtol=2e-6, atol=2e-6)
    assert pool_work["receiver_readout_positive_weight_count"] == 3
    assert edge_work["receiver_readout_positive_weight_count"] == 3
    assert pool_work["receiver_readout_exact_positive_weight_count_available"] == 1


def test_pool_first_erases_weighted_nullspace_that_edge_first_can_express():
    pool = _reader("pool_first").eval()
    edge = _reader("edge_first").eval()
    edge.load_state_dict(pool.state_dict())
    with torch.no_grad():
        pool.query_projection.weight.zero_()
        pool.geometry_projection.weight.zero_()
        pool.output_projection.weight.zero_()
        pool.output_projection.weight[:64, :] = torch.eye(64)
        edge.load_state_dict(pool.state_dict())
    values = torch.zeros(1, 2, 64)
    values[0, 0, 0] = 1.0
    values[0, 1, 0] = -1.0
    prepared = _prepared(values)
    query_states = torch.zeros(1, 1, 128)
    query_coordinates = torch.tensor([[[0.25, 0.4]]])
    anchor_coordinates = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    cover = _cover(
        torch.tensor([[[0, 1]]]),
        torch.tensor([[[31, 32]]]),
        torch.tensor([[[0.5, 0.5]]]),
    )
    pool_value, _, _ = _local(pool, prepared, query_states, query_coordinates, anchor_coordinates, cover)
    edge_value, _, _ = _local(edge, prepared, query_states, query_coordinates, anchor_coordinates, cover)
    torch.testing.assert_close(pool_value, torch.zeros_like(pool_value), rtol=0, atol=0)
    assert edge_value[0, 0, 0] > 0.0


def test_duplicate_anchor_slots_and_anchor_permutation_preserve_local_function():
    reader = _reader("edge_first").eval()
    with torch.no_grad():
        reader.output_projection.weight.normal_(std=0.02)
    values = torch.randn(1, 3, 64)
    ids = torch.tensor([[53, 17, 89]])
    coordinates = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 1.0]]])
    query_states = torch.randn(1, 2, 128)
    query_coordinates = torch.tensor([[[0.2, 0.3], [0.85, 0.9]]])
    query_mask = torch.ones(1, 2, dtype=torch.bool)
    indices = torch.tensor([[[0, 1, 1], [2, 0, 2]]])
    ids_per_slot = ids[:, None, :].expand(1, 2, 3).gather(2, indices)
    weights = torch.tensor([[[0.2, 0.3, 0.5], [0.25, 0.5, 0.25]]])
    cover = _cover(indices, ids_per_slot, weights, query_mask=query_mask)
    original, _, _ = _local(
        reader, _prepared(values, ids), query_states, query_coordinates, coordinates, cover,
        query_mask=query_mask,
    )

    permutation = torch.tensor([2, 0, 1])
    old_to_new = torch.argsort(permutation)
    permuted_values = values.index_select(1, permutation)
    permuted_ids = ids.index_select(1, permutation)
    permuted_coordinates = coordinates.index_select(1, permutation)
    permuted_indices = old_to_new[indices]
    permuted_cover = _cover(permuted_indices, ids_per_slot, weights, query_mask=query_mask)
    permuted, _, _ = _local(
        reader, _prepared(permuted_values, permuted_ids), query_states,
        query_coordinates, permuted_coordinates, permuted_cover, query_mask=query_mask,
    )
    torch.testing.assert_close(permuted, original, rtol=2e-6, atol=2e-6)


def test_empty_query_returns_empty_correction_and_zero_exact_work():
    reader = _reader("edge_first").eval()
    prepared = _prepared(torch.randn(1, 2, 64))
    query_states = torch.empty(1, 0, 128)
    query_coordinates = torch.empty(1, 0, 2)
    anchor_coordinates = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    mask = torch.empty(1, 0, dtype=torch.bool)
    cover = _cover(
        torch.empty(1, 0, 2, dtype=torch.long),
        torch.empty(1, 0, 2, dtype=torch.long),
        torch.empty(1, 0, 2),
        validity=torch.empty(1, 0, 2, dtype=torch.bool),
        query_mask=mask,
    )
    correction, _, work = _local(
        reader, prepared, query_states, query_coordinates, anchor_coordinates,
        cover, query_mask=mask, collect_exact_work_counts=True,
    )
    assert correction.shape == (1, 0, 128)
    assert work["receiver_readout_positive_weight_count"] == 0
    assert work["receiver_readout_query_projection_rows"] == 0


@pytest.mark.parametrize(
    ("weights", "indices", "query_mask", "expected"),
    [
        (torch.tensor([[[-0.1]]]), torch.tensor([[[0]]]), None, "finite and nonnegative"),
        (torch.tensor([[[float("inf")]]]), torch.tensor([[[0]]]), None, "finite and nonnegative"),
        (torch.tensor([[[1.0]]]), torch.tensor([[[2]]]), None, "invalid anchor offset"),
        (torch.tensor([[[0.0]]]), torch.tensor([[[0]]]), torch.tensor([[False]]), "mask.*exactly"),
    ],
)
def test_local_primitive_rejects_invalid_cover_payloads(weights, indices, query_mask, expected):
    reader = _reader("pool_first").eval()
    values = torch.randn(1, 2, 64)
    prepared = _prepared(values)
    actual_ids = torch.tensor([[[31]]])
    cover_mask = None if query_mask is None else torch.tensor([[True]])
    cover = _cover(indices, actual_ids, weights, query_mask=cover_mask)
    query = torch.randn(1, 1, 128)
    qcoords = torch.tensor([[[0.2, 0.2]]])
    ecoords = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    with pytest.raises((ValueError, RuntimeError), match=expected):
        _local(reader, prepared, query, qcoords, ecoords, cover, query_mask=query_mask)


def test_local_primitive_rejects_validity_on_a_masked_native_anchor():
    reader = _reader("pool_first").eval()
    prepared = _prepared(torch.randn(1, 2, 64))
    cover = _cover(torch.tensor([[[1]]]), torch.tensor([[[32]]]), torch.tensor([[[1.0]]]))
    with pytest.raises(ValueError, match="invalid environment anchor"):
        _local(
            reader,
            prepared,
            torch.randn(1, 1, 128),
            torch.tensor([[[0.2, 0.3]]]),
            torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
            cover,
            anchor_mask=torch.tensor([[True, False]]),
        )


def _checkpoint_case(reader, *, checkpointed: bool):
    reader.eval()
    torch.manual_seed(281)
    source_state = torch.randn(1, 2, 128, dtype=torch.float64, requires_grad=True)
    query_states = torch.randn(1, 2, 128, dtype=torch.float64, requires_grad=True)
    query_coordinates = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    anchor_coordinates = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[31, 32]])
    prepared = reader.prepare_scene(
        source_state,
        ids,
        frame_key="readout-frame",
        scene_version="scene-v1",
        owner_token="owner-v1",
    )
    query_mask = torch.ones(1, 2, dtype=torch.bool)
    indices = torch.tensor([[[0, 1], [1, 0]]])
    cover = _cover(
        indices,
        ids[:, None, :].expand(1, 2, 2).gather(2, indices),
        torch.full((1, 2, 2), 0.5, dtype=torch.float64),
        query_mask=query_mask,
    )
    correction, _, work = _local(
        reader, prepared, query_states, query_coordinates, anchor_coordinates,
        cover, query_mask=query_mask, return_trace=not checkpointed,
    )
    watched = (query_states, source_state, query_coordinates, anchor_coordinates, *reader.parameters())
    loss = correction.square().sum() + 0.03 * correction.sum()
    first = torch.autograd.grad(loss, watched, create_graph=True)
    second_scalar = sum(gradient.square().sum() for gradient in first)
    second = torch.autograd.grad(second_scalar, watched)
    return (
        correction.detach(),
        tuple(value.detach() for value in first),
        tuple(value.detach() for value in second),
        work,
    )


def test_nonreentrant_tiled_training_checkpoint_preserves_first_and_second_derivatives():
    checkpointed = copy.deepcopy(_reader("edge_first")).double()
    direct = copy.deepcopy(checkpointed)
    with torch.no_grad():
        checkpointed.output_projection.weight.normal_(std=0.01)
        direct.load_state_dict(checkpointed.state_dict())
    actual = _checkpoint_case(checkpointed, checkpointed=True)
    expected = _checkpoint_case(direct, checkpointed=False)
    for actual_group, expected_group in zip(actual[:3], expected[:3]):
        if isinstance(actual_group, tuple):
            assert len(actual_group) == len(expected_group)
            for left, right in zip(actual_group, expected_group):
                torch.testing.assert_close(left, right, rtol=2e-8, atol=2e-9)
        else:
            torch.testing.assert_close(actual_group, expected_group, rtol=2e-8, atol=2e-9)
    assert actual[3]["receiver_readout_checkpointed_query_tiles"] == 1
    assert actual[3]["receiver_readout_nonlinear_activation_bytes_for_autograd"] == 0
    assert expected[3]["receiver_readout_checkpointed_query_tiles"] == 0
    assert expected[3]["receiver_readout_nonlinear_activation_bytes_for_autograd"] > 0


def test_full_bank_rejects_invalid_native_measures_and_frame_scales():
    reader = _reader(
        "edge_first", geometry_mode="scene_frame_normalized", access="full_bank_diagnostic"
    ).eval()
    prepared = _prepared(torch.randn(1, 2, 64))
    query_states = torch.randn(1, 1, 128)
    query_coordinates = torch.tensor([[[0.25, 0.25]]])
    anchor_coordinates = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    anchor_mask = torch.ones(1, 2, dtype=torch.bool)
    query_mask = torch.ones(1, 1, dtype=torch.bool)
    common = {
        "prepared": prepared,
        "query_states": query_states,
        "query_coordinates": query_coordinates,
        "anchor_coordinates": anchor_coordinates,
        "anchor_mask": anchor_mask,
        "query_mask": query_mask,
        "collect_exact_work_counts": False,
        "frame_scale": torch.ones(1, 2),
    }
    with pytest.raises(ValueError, match="finite positive measure"):
        reader.full_bank_correction(anchor_measure=torch.tensor([[1.0, -0.1]]), **common)
    with pytest.raises(ValueError, match="finite and positive"):
        reader.full_bank_correction(
            anchor_measure=torch.tensor([[1.0, 1.0]]),
            **{**common, "frame_scale": torch.tensor([[1.0, 0.0]])},
        )


def _scene() -> SceneBatch:
    module_coordinates = torch.tensor([[[0.18, 0.20], [0.79, 0.24], [0.48, 0.78]]])
    environment_coordinates = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    )
    return SceneBatch(
        module_features=torch.tensor([[[0.2, -0.1], [0.7, 0.3], [-0.4, 0.6]]]),
        environment_features=torch.tensor([[[0.1, 0.5], [0.6, -0.3], [-0.2, 0.8], [0.4, 0.2]]]),
        module_coordinates=module_coordinates,
        environment_coordinates=environment_coordinates,
        module_mask=torch.ones(1, 3, dtype=torch.bool),
        environment_mask=torch.ones(1, 4, dtype=torch.bool),
        module_measure=torch.tensor([[0.3, 0.4, 0.3]]),
        environment_measure=torch.tensor([[0.2, 0.3, 0.1, 0.4]]),
        module_ids=torch.tensor([[41, 7, 29]]),
        environment_ids=torch.tensor([[83, 19, 61, 37]]),
        local_radius=torch.full((1, 3), 0.48),
        operating_features=torch.tensor([[0.2]]),
        coordinate_origin=torch.zeros(1, 2),
        coordinate_scale=torch.ones(1, 2),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        scene_key="readout-scene",
        normalizer_key="readout-normalizer",
        coordinate_frame_key="readout-frame",
        module_characteristic_lengths=torch.full((1, 3, 2), 0.12),
        environment_characteristic_lengths=torch.full((1, 4, 2), 0.10),
    )


def _model(
    reduction: str,
    *,
    geometry_mode: str = "native_signed",
    exact_counters: bool = False,
) -> HyperedgeReadoutCoreV1:
    return HyperedgeReadoutCoreV1(
        module_input_dim=2,
        environment_input_dim=2,
        coordinate_dim=2,
        query_input_dim=1,
        operating_input_dim=1,
        output_dim=2,
        architecture="IAG-C",
        context_dim=128,
        num_heads=4,
        context_depth=1,
        local_hidden_dim=128,
        dropout=0.0,
        init_seed=29,
        signed_carrier_width=8,
        thermal_response=True,
        normalization="soft",
        scene_executor="dense",
        query_executor="indexed_projected",
        collect_exact_work_counts=exact_counters,
        moment_variant="H",
        moment_width=8,
        processor_hidden_dim=128,
        moment_init_seed=73,
        receiver_reduction=reduction,
        receiver_access="local_cover",
        geometry_mode=geometry_mode,
        receiver_readout_init_seed=119,
    )


def _query(
    scene: SceneBatch,
    coordinates: torch.Tensor | None = None,
    query_ids: torch.Tensor | None = None,
) -> InheritedQueryBatch:
    if coordinates is None:
        coordinates = torch.tensor([[[0.12, 0.16], [0.43, 0.27], [0.77, 0.32], [0.37, 0.82]]])
    batch, count = coordinates.shape[:2]
    if query_ids is None:
        query_ids = torch.arange(700, 700 + count)[None].expand(batch, -1).clone()
    base = QueryBatch(
        coordinates=coordinates,
        features=coordinates[..., :1] - 0.3 * coordinates[..., 1:2],
        query_mask=torch.ones(batch, count, dtype=torch.bool),
        query_ids=query_ids,
        scene_key=scene.scene_key,
        coordinate_frame_key=scene.coordinate_frame_key,
        receiver_key="readout-query",
    )
    cover = smoothstep_grid_receiver_cover(
        coordinates,
        scene.environment_coordinates,
        scene.environment_ids,
        scene.environment_mask,
        base.query_ids,
        frame_key=scene.coordinate_frame_key,
        query_mask=base.query_mask,
        legal_bounds=torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]).expand(batch, -1, -1),
    )
    return with_receiver_cover(base, cover)


def test_h_core_pair_has_identical_birth_config_and_state_dict_layout():
    pool = _model("pool_first").eval()
    edge = _model("edge_first").eval()
    pool_state, edge_state = pool.state_dict(), edge.state_dict()
    assert pool_state.keys() == edge_state.keys()
    for name in pool_state:
        torch.testing.assert_close(pool_state[name], edge_state[name], rtol=0, atol=0)
    assert pool.get_config()["schema"] == "hyperedge_readout_core_v1"
    assert pool.get_config()["geometry_mode"] == "native_signed"
    restored = HyperedgeReadoutCoreV1.from_config(pool.get_config()).eval()
    restored.load_state_dict(pool_state)
    assert restored.get_config() == pool.get_config()
    assert restored.state_dict().keys() == pool_state.keys()
    assert torch.count_nonzero(pool.receiver_readout.output_projection.weight) == 0

    historical_config = {
        key: value for key, value in pool.get_config().items() if key not in {
            "schema", "receiver_reduction", "receiver_access", "geometry_mode",
            "receiver_readout_init_seed",
        }
    }
    historical_config["schema"] = "scene_prepared_moment_hyperedge_core_v1"
    historical = MomentHyperedgeCoreV1.from_config(historical_config)
    common = dict(historical.named_parameters())
    for name, value in common.items():
        torch.testing.assert_close(value, dict(pool.named_parameters())[name], rtol=0, atol=0)


def test_smoothstep_readout_trace_permutation_and_native_input_derivative_lifetime():
    model = _model("edge_first").eval()
    with torch.no_grad():
        model.receiver_readout.output_projection.weight.normal_(std=0.01)
    scene = _scene()
    organization = model.prepare(scene)
    query = _query(scene)
    controls = torch.zeros_like(scene.module_mask, dtype=scene.module_features.dtype)
    encoder_calls = []
    call_handle = model.query_encoder.register_forward_hook(
        lambda _module, _inputs, _output: encoder_calls.append(1)
    )
    ordinary = model.decode(organization, query, controls)
    traced = model.decode(organization, query, controls, return_trace=True)
    call_handle.remove()
    assert len(encoder_calls) == 2
    torch.testing.assert_close(ordinary.nonlinear_field, traced.nonlinear_field, rtol=0, atol=0)
    torch.testing.assert_close(ordinary.thermal_temperature, traced.thermal_temperature, rtol=0, atol=0)
    assert traced.trace["receiver_readout"]["reduction"] == "edge_first"
    assert traced.trace["receiver_readout"]["access"] == "local_cover"
    assert organization.work_counts["receiver_readout_preparation_projection_rows"] == 4
    assert "receiver_readout_positive_weight_count" not in ordinary.work_counts
    assert traced.work_counts["receiver_readout_exact_positive_weight_count_available"] == 0
    assert "receiver_readout_positive_weight_count" not in traced.work_counts

    # Reordering environment tokens and query rows preserves the ID-bound result.
    env_order = torch.tensor([2, 0, 3, 1])
    reordered_scene = SceneBatch(**{
        field: (
            getattr(scene, field).index_select(1, env_order)
            if field in {
                "environment_features", "environment_coordinates", "environment_mask",
                "environment_measure", "environment_ids", "environment_characteristic_lengths",
            }
            else getattr(scene, field)
        )
        for field in scene.__dataclass_fields__
    })
    query_order = torch.tensor([2, 0, 3, 1])
    reordered_query = _query(
        reordered_scene,
        query.coordinates.index_select(1, query_order),
        query.query_ids.index_select(1, query_order),
    )
    reordered_org = model.prepare(reordered_scene)
    reordered = model.decode(reordered_org, reordered_query, controls)
    inverse_query_order = torch.argsort(query_order)
    torch.testing.assert_close(
        reordered.nonlinear_field.index_select(1, inverse_query_order),
        ordinary.nonlinear_field,
        rtol=3e-5,
        atol=3e-6,
    )
    torch.testing.assert_close(
        reordered.thermal_temperature.index_select(1, inverse_query_order),
        ordinary.thermal_temperature,
        rtol=3e-5,
        atol=3e-6,
    )

    # The native higher derivative with respect to query coordinates is a
    # legal probe and must not consume an otherwise reusable frozen scene.
    query_with_grad = _query(scene, query.coordinates.detach().clone().requires_grad_())
    output = model.decode(organization, query_with_grad, controls)
    first = torch.autograd.grad(
        output.thermal_temperature.sum(),
        query_with_grad.coordinates,
        create_graph=True,
    )[0]
    second = torch.autograd.grad(first.square().sum(), query_with_grad.coordinates)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not organization.graph_lifetime.consumed
    reused = model.decode(organization, query, controls)
    reused.nonlinear_field.square().sum().backward()
    assert organization.graph_lifetime.consumed


def test_optional_counter_gate_preserves_receiver_optimizer_and_derivative_results():
    torch.manual_seed(2201)
    counted_off = _model("edge_first", exact_counters=False).eval()
    counted_on = _model("edge_first", exact_counters=True).eval()
    counted_on.load_state_dict(counted_off.state_dict())
    with torch.no_grad():
        counted_off.receiver_readout.output_projection.weight.normal_(std=0.01)
        counted_off.hyperedge_processor.context_projection.weight.fill_(0.001)
        counted_on.load_state_dict(counted_off.state_dict())

    def take_step(model):
        base_scene = _scene()
        scene = SceneBatch(**{
            field: (
                getattr(base_scene, field).clone().requires_grad_()
                if field in {"module_features", "environment_features"}
                else getattr(base_scene, field)
            )
            for field in base_scene.__dataclass_fields__
        })
        query = _query(scene)
        organization = model.prepare(scene)
        controls = torch.tensor([[0.1, -0.2, 0.05]])
        output = model.decode(organization, query, controls)
        loss = output.nonlinear_field.square().sum() + output.thermal_temperature.square().sum()
        watched = (scene.module_features, model.receiver_readout.output_projection.weight,
                   model.field_head.net[2].weight)
        first = torch.autograd.grad(loss, watched, create_graph=True, retain_graph=True)
        second = torch.autograd.grad(
            sum(value.square().sum() for value in first), watched, retain_graph=True
        )
        parameters = tuple(model.parameters())
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        optimizer = torch.optim.SGD(parameters, lr=1.0e-4)
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = None if gradient is None else gradient.detach().clone()
        optimizer.step()
        return (
            output.nonlinear_field.detach(),
            output.thermal_temperature.detach(),
            tuple(value.detach() for value in first),
            tuple(value.detach() for value in second),
            {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
            organization.work_counts,
        )

    off = take_step(counted_off)
    on = take_step(counted_on)
    torch.testing.assert_close(off[0], on[0], rtol=0, atol=0)
    torch.testing.assert_close(off[1], on[1], rtol=0, atol=0)
    for left_group, right_group in zip(off[2:4], on[2:4]):
        for left, right in zip(left_group, right_group):
            torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-7)
    assert off[5]["atlas_exact_counts_available"] == 0
    assert on[5]["atlas_exact_counts_available"] == 1
    assert "module_positive_associations" not in off[5]
    assert "module_positive_associations" in on[5]
    for name in off[4]:
        torch.testing.assert_close(off[4][name], on[4][name], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("stale_part", "expected"),
    [
        ("projection", "Prepared scene/atlas/value factors changed"),
        ("weights", "Model weights or normalization buffers changed"),
        ("buffer", "Model weights or normalization buffers changed"),
        ("geometry", "Prepared scene/atlas/value factors changed"),
    ],
)
def test_readout_rejects_stale_preparation_factors_and_scene_geometry(stale_part, expected):
    model = _model("pool_first").eval()
    scene = _scene()
    organization = model.prepare(scene)
    query = _query(scene)
    assert organization.receiver_readout_values.source_state is organization.moment_hyperedges.processed_state
    assert organization.receiver_readout_values.anchor_ids is scene.environment_ids
    with torch.no_grad():
        if stale_part == "projection":
            organization.receiver_readout_values.values.add_(0.01)
        elif stale_part == "weights":
            model.receiver_readout.query_projection.weight.add_(0.01)
        elif stale_part == "buffer":
            model.normalization_epoch.add_(1)
        else:
            scene.environment_coordinates.add_(0.01)
    with pytest.raises(ValueError, match=expected):
        model.decode(organization, query)
