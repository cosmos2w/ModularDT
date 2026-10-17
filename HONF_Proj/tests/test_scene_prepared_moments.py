from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from honf_forward_core.content_hypergraph.contracts import QueryBatch, SceneBatch
from honf_forward_core.inherited_attention.hyperedge_processor import MomentHyperedgeProcessor
from honf_forward_core.inherited_attention.moment_model import MomentHyperedgeCoreV1
from honf_forward_core.inherited_attention.moments import (
    MomentGraphLifetime,
    PreparedMomentHyperedges,
    build_moment_descriptor,
    compute_typed_moments,
    export_prepared_moment_hyperedges,
)
from honf_forward_core.inherited_attention.normalization import measure_relative_softmax


def _case(*, dtype: torch.dtype = torch.float64, coordinate_dim: int = 2, requires_grad: bool = False):
    generator = torch.Generator(device="cpu").manual_seed(1937 + coordinate_dim)
    batch, heads, anchors, donors = 2, 4, 3, 5
    values = torch.randn(batch, heads, donors, 32, generator=generator, dtype=dtype)
    lift = torch.randn(heads, 8, 32, generator=generator, dtype=dtype)
    logits = torch.randn(batch, heads, anchors, donors, generator=generator, dtype=dtype)
    probabilities = logits.softmax(dim=-1)
    donor_valid = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    anchor_valid = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    probabilities = probabilities * donor_valid[:, None, None, :].to(dtype)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    probabilities = probabilities * anchor_valid[:, None, :, None].to(dtype)
    # Preserve a real row-mass drift case rather than silently renormalizing it.
    probabilities = probabilities.clone()
    probabilities[0, :, 0] *= 1.0 + (2.0e-6 if dtype == torch.float32 else 2.0e-12)
    donor_coordinates = torch.randn(batch, donors, coordinate_dim, generator=generator, dtype=dtype)
    anchor_coordinates = torch.randn(batch, anchors, coordinate_dim, generator=generator, dtype=dtype)
    donor_lengths = 0.2 + torch.rand(batch, donors, coordinate_dim, generator=generator, dtype=dtype)
    donor_ids = torch.tensor([[11, 12, 13, 14, -1], [21, 22, 23, 24, 25]], dtype=torch.int64)
    anchor_ids = torch.tensor([[31, 32, -1], [41, 42, 43]], dtype=torch.int64)
    if requires_grad:
        for tensor in (values, lift, probabilities, donor_coordinates, anchor_coordinates, donor_lengths):
            tensor.requires_grad_()
    full_mean = torch.einsum("bhen,bhnd->bhed", probabilities, values)
    return {
        "values": values,
        "lift_weight": lift,
        "probabilities": probabilities,
        "full_value_mean": full_mean,
        "normalized_donor_coordinates": donor_coordinates,
        "normalized_anchor_coordinates": anchor_coordinates,
        "normalized_donor_characteristic_lengths": donor_lengths,
        "donor_valid": donor_valid,
        "anchor_valid": anchor_valid,
        "donor_ids": donor_ids,
        "anchor_ids": anchor_ids,
    }


def _compute(case, kind="module"):
    return compute_typed_moments(kind=kind, **case)


@pytest.mark.parametrize("coordinate_dim", [2, 3])
def test_typed_moments_match_independent_fp64_literal_and_descriptor_order(coordinate_dim):
    case = _case(coordinate_dim=coordinate_dim)
    stats = _compute(case)
    p = case["probabilities"] * (
        case["anchor_valid"][:, None, :, None] & case["donor_valid"][:, None, None, :]
    ).to(torch.float64)
    values, lift = case["values"], case["lift_weight"]
    xi, xi_anchor = case["normalized_donor_coordinates"], case["normalized_anchor_coordinates"]
    u = torch.einsum("bhnd,hkd->bhnk", values, lift)
    relative = xi[:, None, None, :, :] - xi_anchor[:, None, :, None, :]
    row_mass = p.sum(dim=-1)
    mean_u = torch.einsum("bhen,bhnk->bhek", p, u)
    centroid = torch.einsum("bhen,bhend->bhed", p, relative)
    cross = torch.einsum("bhen,bhend,bhnk->bhedk", p, relative, u)
    covariance = torch.einsum("bhen,bhena,bhenc->bheac", p, relative, relative)
    covariance = covariance - centroid[..., :, None] * centroid[..., None, :]
    variance = torch.einsum("bhen,bhnk->bhek", p, u.square()) - mean_u.square()
    weighted_length = torch.einsum("bhen,bnd->bhed", p, case["normalized_donor_characteristic_lengths"])
    typed_valid = (row_mass > 0) & case["anchor_valid"][:, None, :]
    upper = torch.triu_indices(coordinate_dim, coordinate_dim)

    torch.testing.assert_close(stats.row_mass, row_mass)
    torch.testing.assert_close(stats.lifted_value_mean, torch.where(typed_valid[..., None], mean_u, 0.0))
    torch.testing.assert_close(stats.relative_centroid, torch.where(typed_valid[..., None], centroid, 0.0))
    torch.testing.assert_close(stats.position_value_cross,
                               torch.where(typed_valid[..., None, None], cross, 0.0))
    torch.testing.assert_close(stats.spatial_covariance_upper,
                               torch.where(typed_valid[..., None], covariance[..., upper[0], upper[1]], 0.0))
    torch.testing.assert_close(stats.latent_variance, torch.where(typed_valid[..., None], variance, 0.0))
    torch.testing.assert_close(stats.normalized_donor_length,
                               torch.where(typed_valid[..., None], weighted_length, 0.0))
    positive_count = (p > 0).sum(dim=-1)
    torch.testing.assert_close(stats.audit.positive_donor_count, positive_count)
    torch.testing.assert_close(stats.audit.gamma_order, 4 * positive_count + 16)
    assert stats.audit.epsilon == torch.finfo(torch.float64).eps

    anchor_length = 0.3 + torch.rand(2, 3, coordinate_dim, generator=torch.Generator().manual_seed(9),
                                     dtype=torch.float64)
    operating = torch.randn(2, 128, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
    descriptor = build_moment_descriptor(
        stats,
        _compute(case, kind="environment"),
        normalized_anchor_coordinates=xi_anchor,
        normalized_anchor_characteristic_lengths=anchor_length,
        operating_state=operating,
    )
    expected_width = 708 if coordinate_dim == 2 else 814
    assert descriptor.shape == (2, 3, expected_width)
    torch.testing.assert_close(descriptor[..., :coordinate_dim], xi_anchor)
    torch.testing.assert_close(descriptor[..., coordinate_dim:2 * coordinate_dim], anchor_length)
    torch.testing.assert_close(descriptor[..., 2 * coordinate_dim:2 * coordinate_dim + 128],
                               operating[:, None, :].expand(2, 3, 128))


def test_fp32_row_mass_drift_roundoff_is_clamped_and_large_negative_is_flagged():
    case = _case(dtype=torch.float32, coordinate_dim=2)
    # A constant lifted donor has zero normalized variance.  The deliberately
    # shifted P mass creates only the known s(1-s)c^2 literal drift.
    values = torch.zeros_like(case["values"])
    values[..., 0] = 1.0e4
    lift = torch.zeros_like(case["lift_weight"])
    lift[..., 0] = 1.0
    case["values"] = values
    case["lift_weight"] = lift
    case["full_value_mean"] = torch.einsum("bhen,bhnd->bhed", case["probabilities"], values)
    stats = _compute(case)
    assert stats.audit.gamma_order[0, 0, 0] == 4 * stats.audit.positive_donor_count[0, 0, 0] + 16
    assert bool(stats.audit.latent_clamped[0, 0, 0, 0])
    assert not bool(stats.audit.latent_large_negative[0, 0, 0, 0])
    assert stats.latent_variance[0, 0, 0, 0] == 0
    assert stats.audit.latent_allowance[0, 0, 0, 0] > 0

    bad = dict(case)
    bad["probabilities"] = case["probabilities"].clone()
    bad["probabilities"][0, :, 0] = bad["probabilities"][0, :, 0] * 1.01
    bad["full_value_mean"] = torch.einsum("bhen,bhnd->bhed", bad["probabilities"], values)
    bad_stats = _compute(bad)
    assert bool(bad_stats.audit.latent_large_negative[0, 0, 0, 0])
    assert bad_stats.audit.latent_raw_variance[0, 0, 0, 0] < -bad_stats.audit.latent_allowance[0, 0, 0, 0]


def test_relative_moments_are_invariant_to_a_shared_coordinate_translation():
    case = _case(coordinate_dim=3)
    original = _compute(case, kind="environment")
    shifted = dict(case)
    translation = torch.tensor([1024.0, -2048.0, 512.0], dtype=torch.float64)
    shifted["normalized_donor_coordinates"] = case["normalized_donor_coordinates"] + translation
    shifted["normalized_anchor_coordinates"] = case["normalized_anchor_coordinates"] + translation
    transformed = _compute(shifted, kind="environment")

    for name in (
        "full_value_mean", "lifted_value_mean", "relative_centroid", "position_value_cross",
        "spatial_covariance_upper", "latent_variance", "normalized_donor_length", "row_mass",
    ):
        torch.testing.assert_close(getattr(transformed, name), getattr(original, name), rtol=2e-11, atol=2e-10)
    for name in (
        "latent_raw_variance", "latent_raw_scale", "latent_allowance", "spatial_raw_diagonal",
        "spatial_raw_scale", "spatial_allowance",
    ):
        torch.testing.assert_close(getattr(transformed.audit, name), getattr(original.audit, name),
                                   rtol=2e-11, atol=2e-10)


def test_identical_environment_token_measure_split_preserves_typed_moments():
    case = _case(coordinate_dim=2)
    batch, heads, donors, _ = case["values"].shape
    generator = torch.Generator(device="cpu").manual_seed(4081)
    logits = torch.randn(batch, heads, case["probabilities"].shape[2], donors,
                         generator=generator, dtype=torch.float64)
    measure = torch.tensor([[0.3, 0.7, 0.4, 0.9, 0.0], [0.6, 0.2, 0.8, 0.5, 0.1]], dtype=torch.float64)
    valid = case["donor_valid"]
    destination_valid = case["anchor_valid"]
    probabilities = measure_relative_softmax(logits, measure, valid, destination_valid)

    split_index = 2

    def split_donor_axis(tensor: torch.Tensor, axis: int) -> torch.Tensor:
        before = tensor.narrow(axis, 0, split_index)
        donor = tensor.narrow(axis, split_index, 1)
        after = tensor.narrow(axis, split_index + 1, tensor.shape[axis] - split_index - 1)
        return torch.cat((before, donor, donor, after), dim=axis)

    split_measure = torch.cat((measure[:, :split_index], measure[:, split_index:split_index + 1] / 2,
                               measure[:, split_index:split_index + 1] / 2, measure[:, split_index + 1:]), dim=1)
    split_valid = split_donor_axis(valid, 1)
    split_logits = split_donor_axis(logits, 3)
    split_probabilities = measure_relative_softmax(
        split_logits, split_measure, split_valid, destination_valid
    )
    merged_probabilities = torch.cat((split_probabilities[..., :split_index],
                                      split_probabilities[..., split_index:split_index + 2].sum(dim=-1, keepdim=True),
                                      split_probabilities[..., split_index + 2:]), dim=-1)
    torch.testing.assert_close(merged_probabilities, probabilities, rtol=2e-12, atol=2e-12)

    original_case = dict(case)
    original_case["probabilities"] = probabilities
    original_case["full_value_mean"] = torch.einsum("bhen,bhnd->bhed", probabilities, case["values"])
    split_case = dict(case)
    split_case["values"] = split_donor_axis(case["values"], 2)
    split_case["probabilities"] = split_probabilities
    split_case["full_value_mean"] = torch.einsum("bhen,bhnd->bhed", split_probabilities, split_case["values"])
    split_case["normalized_donor_coordinates"] = split_donor_axis(
        case["normalized_donor_coordinates"], 1
    )
    split_case["normalized_donor_characteristic_lengths"] = split_donor_axis(
        case["normalized_donor_characteristic_lengths"], 1
    )
    split_case["donor_valid"] = split_valid
    split_ids = split_donor_axis(case["donor_ids"], 1)
    split_ids[:, split_index + 1] += 10_000
    split_case["donor_ids"] = split_ids

    original = _compute(original_case, kind="environment")
    split = _compute(split_case, kind="environment")
    for name in (
        "full_value_mean", "lifted_value_mean", "relative_centroid", "position_value_cross",
        "spatial_covariance_upper", "latent_variance", "normalized_donor_length", "typed_valid", "row_mass",
    ):
        torch.testing.assert_close(getattr(split, name), getattr(original, name), rtol=2e-12, atol=2e-12)


def test_moment_primitives_pass_double_backward_finite_difference_checks():
    case = _case(coordinate_dim=2)
    # Keep finite-difference evaluation bounded while retaining all four heads
    # and the native width-32 donor states.
    case["values"] = case["values"][:1, :, :2].requires_grad_()
    case["lift_weight"] = case["lift_weight"].detach()
    case["probabilities"] = case["probabilities"][:1, :, :1, :2].requires_grad_()
    case["normalized_donor_coordinates"] = case["normalized_donor_coordinates"][:1, :2].requires_grad_()
    case["normalized_anchor_coordinates"] = case["normalized_anchor_coordinates"][:1, :1].requires_grad_()
    case["normalized_donor_characteristic_lengths"] = (
        case["normalized_donor_characteristic_lengths"][:1, :2].requires_grad_()
    )
    case["donor_valid"] = case["donor_valid"][:1, :2]
    case["anchor_valid"] = case["anchor_valid"][:1, :1]
    case["donor_ids"] = case["donor_ids"][:1, :2]
    case["anchor_ids"] = case["anchor_ids"][:1, :1]

    def outputs(values, lift, p, xi, xi_anchor, length):
        mean = torch.einsum("bhen,bhnd->bhed", p, values)
        result = compute_typed_moments(
            kind="module",
            values=values,
            full_value_mean=mean,
            probabilities=p,
            lift_weight=lift,
            normalized_donor_coordinates=xi,
            normalized_anchor_coordinates=xi_anchor,
            normalized_donor_characteristic_lengths=length,
            donor_valid=case["donor_valid"],
            anchor_valid=case["anchor_valid"],
            donor_ids=case["donor_ids"],
            anchor_ids=case["anchor_ids"],
        )
        return (result.lifted_value_mean, result.relative_centroid, result.position_value_cross,
                result.spatial_covariance_upper, result.latent_variance,
                result.normalized_donor_length, result.full_value_mean)

    inputs = (case["values"], case["lift_weight"], case["probabilities"],
              case["normalized_donor_coordinates"], case["normalized_anchor_coordinates"],
              case["normalized_donor_characteristic_lengths"])
    assert torch.autograd.gradcheck(outputs, inputs, eps=1e-6, atol=2e-5, rtol=2e-4, fast_mode=True)
    assert torch.autograd.gradgradcheck(outputs, inputs, eps=1e-6, atol=3e-5, rtol=3e-4, fast_mode=True)


def test_m_and_h_processors_have_identical_inventory_and_zero_output_attachment():
    m = MomentHyperedgeProcessor(708, 2, processor="M", thermal_transport=True, init_seed=73)
    h = MomentHyperedgeProcessor(708, 2, processor="H", thermal_transport=True, init_seed=73)
    assert tuple(name for name, _ in m.named_parameters()) == tuple(name for name, _ in h.named_parameters())
    for (name_m, parameter_m), (name_h, parameter_h) in zip(m.named_parameters(), h.named_parameters()):
        assert name_m == name_h
        assert parameter_m.shape == parameter_h.shape
        torch.testing.assert_close(parameter_m, parameter_h, rtol=0, atol=0)
    assert torch.count_nonzero(m.input_projection.weight) > 0
    assert torch.count_nonzero(m.context_projection.weight) == 0
    assert torch.count_nonzero(m.context_projection.bias) == 0
    assert torch.count_nonzero(m.heat_constant_projection.weight) == 0
    assert torch.count_nonzero(m.heat_spatial_projection.weight) == 0

    descriptor = torch.randn(2, 3, 708, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
    out_m = m(descriptor, anchor_mask=mask)
    out_h = h(descriptor, anchor_mask=mask)
    assert out_m.context_residual.shape == (2, 3, 128)
    assert out_m.heat_mix_constant.shape == (2, 4, 3, 8)
    assert out_m.heat_mix_spatial.shape == (2, 4, 3, 8, 2)
    assert torch.count_nonzero(out_m.context_residual) == 0
    assert torch.count_nonzero(out_h.heat_mix_constant) == 0
    assert torch.count_nonzero(out_h.heat_mix_spatial) == 0
    (out_m.context_residual.sum() + out_m.heat_mix_constant.sum() + out_m.heat_mix_spatial.sum()).backward()
    assert m.context_projection.weight.grad is not None
    assert torch.count_nonzero(m.context_projection.weight.grad) > 0
    assert m.input_projection.weight.grad is not None
    assert torch.count_nonzero(m.input_projection.weight.grad) == 0

    with torch.no_grad():
        m.context_projection.weight.fill_(0.01)
    active = m(descriptor.detach().requires_grad_(), anchor_mask=mask)
    active.context_residual.sum().backward()
    assert torch.count_nonzero(m.input_projection.weight.grad) > 0
    assert torch.count_nonzero(active.context_residual[~mask]) == 0


def _prepared(*, lifetime=None):
    case = _case(dtype=torch.float64, coordinate_dim=2)
    module = _compute(case, kind="module")
    environment = _compute(case, kind="environment")
    descriptor = torch.randn(2, 3, 708, dtype=torch.float64, requires_grad=True)
    base = torch.randn(2, 3, 128, dtype=torch.float64, requires_grad=True)
    module_coordinates = case["normalized_donor_coordinates"]
    environment_coordinates = case["normalized_anchor_coordinates"]
    return PreparedMomentHyperedges(
        module=module,
        environment=environment,
        descriptor=descriptor,
        base_projected_state=base,
        module_measure=torch.ones(2, 5, dtype=torch.float64),
        source_carriers=torch.ones(2, 4, 5, 8, dtype=torch.float64),
        module_coordinates=module_coordinates,
        environment_coordinates=environment_coordinates,
        module_ids=case["donor_ids"],
        environment_ids=case["anchor_ids"],
        module_mask=case["donor_valid"],
        environment_mask=case["anchor_valid"],
        source_module_coordinates=case["normalized_donor_coordinates"],
        source_environment_coordinates=case["normalized_anchor_coordinates"],
        frame_key="fixture_frame",
        scene_version=(4, 9),
        owner_token=object(),
        normalizer_key="normalizer_1",
        scene_key="scene_17",
        graph_lifetime=lifetime,
    )


def test_prepared_view_is_same_cache_and_cache_guards_cover_mutation_and_consumed_graph():
    lifetime = MomentGraphLifetime()
    prepared = _prepared(lifetime=lifetime)
    view = export_prepared_moment_hyperedges(prepared)
    assert view.prepared is prepared
    assert view.descriptor is prepared.descriptor
    prepared.validate_binding(
        frame_key="fixture_frame",
        scene_version=(4, 9),
        owner_token=prepared.owner_token,
        normalizer_key="normalizer_1",
        scene_key="scene_17",
        module_ids=prepared.module_ids,
        environment_ids=prepared.environment_ids,
        module_coordinates=prepared.module_coordinates,
        environment_coordinates=prepared.environment_coordinates,
    )
    with torch.no_grad():
        prepared.descriptor.add_(1.0)
    with pytest.raises(ValueError, match="changed after construction"):
        view.validate_unchanged()

    prepared = _prepared()
    with torch.no_grad():
        prepared.source_module_coordinates.add_(0.25)
    with pytest.raises(ValueError, match="changed after construction"):
        prepared.validate_binding(module_coordinates=prepared.source_module_coordinates)

    lifetime = MomentGraphLifetime()
    prepared = _prepared(lifetime=lifetime)
    prepared = prepared.with_processor_outputs(
        preactivation=prepared.descriptor[..., :128],
        hidden=prepared.descriptor[..., :128].tanh(),
        context_residual=prepared.base_projected_state.square(),
        processed_state=prepared.base_projected_state + prepared.base_projected_state.square(),
        heat_mix_constant=None,
        heat_mix_spatial=None,
    )
    (prepared.processed_state.sum() + prepared.module.latent_variance.sum()).backward()
    assert lifetime.consumed
    with pytest.raises(ValueError, match="consumed"):
        prepared.validate_unchanged()


def _scene_batch() -> SceneBatch:
    return SceneBatch(
        module_features=torch.tensor([[[0.2, -0.1], [0.7, 0.3], [-0.4, 0.6]]]),
        environment_features=torch.tensor([[[0.1, 0.5], [0.6, -0.3], [-0.2, 0.8], [0.4, 0.2]]]),
        module_coordinates=torch.tensor([[[0.18, 0.20], [0.79, 0.24], [0.48, 0.78]]]),
        environment_coordinates=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]),
        module_mask=torch.ones(1, 3, dtype=torch.bool),
        environment_mask=torch.ones(1, 4, dtype=torch.bool),
        module_measure=torch.tensor([[0.3, 0.4, 0.3]]),
        environment_measure=torch.tensor([[0.2, 0.3, 0.1, 0.4]]),
        module_ids=torch.tensor([[41, 7, 29]], dtype=torch.long),
        environment_ids=torch.tensor([[83, 19, 61, 37]], dtype=torch.long),
        local_radius=torch.full((1, 3), 0.48),
        operating_features=torch.tensor([[0.2]]),
        coordinate_origin=torch.zeros(1, 2),
        coordinate_scale=torch.ones(1, 2),
        module_measure_reference=torch.ones(1),
        environment_measure_reference=torch.ones(1),
        scene_key="moment-scene-fixture",
        normalizer_key="moment-input-transform",
        coordinate_frame_key="moment-frame-2d",
        module_characteristic_lengths=torch.full((1, 3, 2), 0.12),
        environment_characteristic_lengths=torch.full((1, 4, 2), 0.10),
    )


def _moment_core(
    variant: str,
    *,
    coordinate_dim: int = 2,
    thermal: bool = True,
    collect_exact_work_counts: bool = False,
):
    return MomentHyperedgeCoreV1(
        module_input_dim=2,
        environment_input_dim=2,
        coordinate_dim=coordinate_dim,
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
        thermal_response=thermal,
        normalization="soft",
        scene_executor="dense",
        query_executor="indexed_projected",
        collect_exact_work_counts=collect_exact_work_counts,
        moment_variant=variant,
        moment_init_seed=73,
    )


def test_environment_donor_and_receiver_permutation_preserves_id_bound_hyperedges():
    model = _moment_core("M", thermal=False)
    model.eval()
    scene = _scene_batch()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    permuted_scene = replace(
        scene,
        environment_features=scene.environment_features.index_select(1, permutation),
        environment_coordinates=scene.environment_coordinates.index_select(1, permutation),
        environment_mask=scene.environment_mask.index_select(1, permutation),
        environment_measure=scene.environment_measure.index_select(1, permutation),
        environment_ids=scene.environment_ids.index_select(1, permutation),
        environment_characteristic_lengths=scene.environment_characteristic_lengths.index_select(1, permutation),
    )
    with torch.no_grad():
        original = model.prepare(scene)
        permuted = model.prepare(permuted_scene)
        model.validate_prepared_state(original)
        model.validate_prepared_state(permuted)

        assert torch.equal(permuted.environment_atlas.anchor_ids.index_select(1, inverse),
                           original.environment_atlas.anchor_ids)
        torch.testing.assert_close(
            permuted.module_atlas.probabilities.index_select(2, inverse),
            original.module_atlas.probabilities,
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            permuted.environment_atlas.probabilities.index_select(2, inverse).index_select(3, inverse),
            original.environment_atlas.probabilities,
            rtol=2e-5,
            atol=2e-6,
        )

        for name in (
            "full_value_mean", "lifted_value_mean", "relative_centroid", "position_value_cross",
            "spatial_covariance_upper", "latent_variance", "normalized_donor_length", "typed_valid", "row_mass",
        ):
            torch.testing.assert_close(
                getattr(permuted.moment_hyperedges.module, name).index_select(2, inverse),
                getattr(original.moment_hyperedges.module, name),
                rtol=2e-5,
                atol=2e-6,
            )
            torch.testing.assert_close(
                getattr(permuted.moment_hyperedges.environment, name).index_select(2, inverse),
                getattr(original.moment_hyperedges.environment, name),
                rtol=2e-5,
                atol=2e-6,
            )
        for name in ("descriptor", "processed_state", "context_residual"):
            torch.testing.assert_close(
                getattr(permuted.moment_hyperedges, name).index_select(1, inverse),
                getattr(original.moment_hyperedges, name),
                rtol=2e-5,
                atol=2e-6,
            )


def test_moment_core_schema_preparation_recovery_and_new_organization_intervention():
    c0, m, h = (
        _moment_core("C0"),
        _moment_core("M", collect_exact_work_counts=True),
        _moment_core("H"),
    )
    c0.eval()
    m.eval()
    h.eval()
    assert c0.get_config()["schema"] == "scene_prepared_moment_hyperedge_core_v1"
    assert m.get_config()["moment_variant"] == "M"
    assert h.get_config()["moment_variant"] == "H"
    assert MomentHyperedgeCoreV1.from_config(m.get_config()).get_config() == m.get_config()
    m_parameters, h_parameters = dict(m.named_parameters()), dict(h.named_parameters())
    assert m_parameters.keys() == h_parameters.keys()
    for name, value in m_parameters.items():
        torch.testing.assert_close(value, h_parameters[name], rtol=0, atol=0)
    common_c0 = dict(c0.named_parameters())
    for name, value in common_c0.items():
        torch.testing.assert_close(value, m_parameters[name], rtol=0, atol=0)
    assert len(set(m_parameters) - set(common_c0)) > 0
    assert len(set(h_parameters) - set(common_c0)) == len(set(m_parameters) - set(common_c0))

    with torch.no_grad():
        scene = _scene_batch()
        c0_org = c0.prepare(scene)
        m_initial_org = m.prepare(scene)
        h_initial_org = h.prepare(scene)
        torch.testing.assert_close(m_initial_org.prepared_readout_values.values,
                                   c0_org.prepared_readout_values.values, rtol=0, atol=0)
        torch.testing.assert_close(h_initial_org.prepared_readout_values.values,
                                   c0_org.prepared_readout_values.values, rtol=0, atol=0)

        # Exercise the zero-output counterfactual after a nonzero attachment.
        m.hyperedge_processor.context_projection.weight.fill_(0.002)
        m.hyperedge_processor.heat_constant_projection.weight.fill_(0.001)
        m.hyperedge_processor.heat_spatial_projection.weight.fill_(0.001)
        organization = m.prepare(scene)
        m.validate_prepared_state(organization)
        cache = organization.moment_hyperedges
        assert cache.descriptor.shape == (1, 4, 708)
        assert cache.processed_state.shape == (1, 4, 128)
        assert cache.heat_mix_constant.shape == (1, 4, 4, 8)
        assert cache.heat_mix_spatial.shape == (1, 4, 4, 8, 2)
        torch.testing.assert_close(organization.module_atlas.normalized_measure,
                                   torch.tensor([[0.3, 0.4, 0.3]]))
        torch.testing.assert_close(organization.environment_atlas.normalized_measure,
                                   torch.tensor([[0.2, 0.3, 0.1, 0.4]]))
        torch.testing.assert_close(
            cache.module.normalized_donor_length,
            torch.einsum("bhen,bnd->bhed", organization.module_atlas.probabilities,
                         scene.module_characteristic_lengths),
        )
        torch.testing.assert_close(
            cache.environment.normalized_donor_length,
            torch.einsum("bhen,bnd->bhed", organization.environment_atlas.probabilities,
                         scene.environment_characteristic_lengths),
        )
        assert organization.prepared_readout_values.values is cache.processed_state
        view = m.export_prepared_moment_hyperedges(organization)
        assert view.prepared is cache
        work = organization.work_counts
        assert work["moment_typed_lifted_donor_rows"] == 4 * (scene.num_modules + scene.num_environment_tokens)
        assert work["moment_donor_lift_channels_per_row"] == 56
        assert work["moment_typed_lifted_donor_scalar_entries"] == work["moment_typed_lifted_donor_rows"] * 56
        assert work["moment_processed_anchor_rows"] == scene.batch_size * scene.num_environment_tokens
        assert work["moment_descriptor_width"] == 708
        assert work["thermal_source_response_factor_entries_per_application"] == (
            4 * scene.num_environment_tokens * 8 * scene.num_modules
        )
        assert work["thermal_source_response_factor_materialized_bytes_fp32_per_application"] == (
            work["thermal_source_response_factor_entries_per_application"] * 4
        )
        assert work["thermal_source_response_factor_retained_bytes"] == 0
        assert work["moment_cache_unique_storage_bytes"] >= work["moment_cache_incremental_storage_bytes"]
        assert work["moment_cache_unique_storage_count"] >= work["moment_cache_incremental_storage_count"]
        logical_cache_bytes = sum(tensor.untyped_storage().nbytes() for tensor in cache.tensors())
        assert work["moment_cache_unique_storage_bytes"] <= logical_cache_bytes

        zeroed = m.zero_new_outputs_for_diagnostic(organization)
        assert zeroed is not organization
        assert zeroed.moment_hyperedges is not cache
        torch.testing.assert_close(zeroed.moment_hyperedges.processed_state,
                                   zeroed.moment_hyperedges.base_projected_state)
        assert torch.count_nonzero(zeroed.moment_hyperedges.context_residual) == 0
        assert torch.count_nonzero(zeroed.moment_hyperedges.heat_mix_constant) == 0
        assert torch.count_nonzero(zeroed.moment_hyperedges.heat_mix_spatial) == 0
        assert torch.count_nonzero(cache.context_residual) > 0


        module_p = organization.module_atlas.probabilities.clone()
        module_p[:, :, 0, :] = 0.0
        intervened = m.intervene_associations(
            organization,
            module_p,
            organization.environment_atlas.probabilities.clone(),
            label="empty-module-row-fixture",
        )
        m.validate_prepared_state(intervened)
        assert intervened.moment_hyperedges.module.probabilities is intervened.module_atlas.probabilities
        assert not bool(intervened.moment_hyperedges.module.typed_valid[:, :, 0].any())
        assert torch.count_nonzero(intervened.moment_hyperedges.module.lifted_value_mean[:, :, 0]) == 0
        assert organization.moment_hyperedges is cache


def test_optional_counter_gate_preserves_h_outputs_and_first_second_derivatives():
    from honf_forward_core.inherited_attention import smoothstep_grid_receiver_cover, with_receiver_cover

    def evaluate(exact_counters: bool):
        base_scene = _scene_batch()
        scene = replace(
            base_scene,
            module_features=base_scene.module_features.clone().requires_grad_(),
            environment_features=base_scene.environment_features.clone().requires_grad_(),
            environment_coordinates=base_scene.environment_coordinates.clone().requires_grad_(),
        )
        model = _moment_core(
            "H", thermal=False, collect_exact_work_counts=exact_counters
        ).eval()
        coordinates = torch.tensor([[[0.18, 0.21], [0.82, 0.75]]], requires_grad=True)
        base_query = QueryBatch(
            coordinates=coordinates,
            features=torch.tensor([[[0.1], [-0.2]]]),
            query_mask=torch.ones(1, 2, dtype=torch.bool),
            query_ids=torch.tensor([[501, 502]]),
            scene_key=scene.scene_key,
            coordinate_frame_key=scene.coordinate_frame_key,
            receiver_key="counter-equivalence-query",
        )
        cover = smoothstep_grid_receiver_cover(
            coordinates,
            scene.environment_coordinates,
            scene.environment_ids,
            scene.environment_mask,
            base_query.query_ids,
            frame_key=scene.coordinate_frame_key,
            query_mask=base_query.query_mask,
        )
        query = with_receiver_cover(base_query, cover)
        organization = model.prepare(scene)
        output = model.decode(organization, query)
        loss = output.nonlinear_field.square().sum() + 0.03 * output.nonlinear_field.sum()
        watched = (
            scene.module_features,
            model.field_head.net[2].weight,
        )
        first = torch.autograd.grad(loss, watched, create_graph=True)
        second_loss = sum(value.square().sum() for value in first)
        second = torch.autograd.grad(second_loss, watched)
        return output.nonlinear_field.detach(), first, second, organization.work_counts

    without = evaluate(False)
    with_counts = evaluate(True)
    torch.testing.assert_close(without[0], with_counts[0], rtol=0, atol=0)
    for left_group, right_group in zip(without[1:3], with_counts[1:3]):
        for left, right in zip(left_group, right_group):
            torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-7)
    assert without[3]["atlas_exact_counts_available"] == 0
    assert without[3]["moment_optional_counters_available"] == 0
    for key in (
        "atlas_eligible_associations",
        "atlas_positive_associations",
        "module_positive_associations",
        "moment_cache_logical_tensor_count",
        "moment_cache_unique_storage_bytes",
        "moment_latent_variance_clamped_entries",
    ):
        assert key not in without[3]
    assert with_counts[3]["atlas_exact_counts_available"] == 1
    assert with_counts[3]["moment_optional_counters_available"] == 1
    assert "atlas_positive_associations" in with_counts[3]
    assert "moment_cache_unique_storage_bytes" in with_counts[3]


def test_actual_moment_core_rejects_stale_owner_frame_ids_parameters_and_consumed_graph():
    model = _moment_core("M", thermal=False)
    model.eval()
    scene = _scene_batch()

    with torch.no_grad():
        prepared = model.prepare(scene)
        other_model = _moment_core("M", thermal=False)
        other_model.eval()
        with pytest.raises(ValueError, match="belongs to another model"):
            other_model.validate_prepared_state(prepared)

        stale_frame = replace(prepared.moment_hyperedges, frame_key="stale-frame")
        prepared.moment_hyperedges = stale_frame
        prepared.factor_versions = model._factors(prepared)
        with pytest.raises(ValueError, match="different frame binding"):
            model.validate_prepared_state(prepared)

        prepared = model.prepare(scene)
        stale_ids = replace(prepared.moment_hyperedges, module_ids=scene.module_ids.clone())
        prepared.moment_hyperedges = stale_ids
        prepared.factor_versions = model._factors(prepared)
        with pytest.raises(ValueError, match="different module IDs tensor objects"):
            model.validate_prepared_state(prepared)

        prepared = model.prepare(scene)
        with torch.no_grad():
            model.hyperedge_processor.context_projection.bias.add_(0.01)
        with pytest.raises(ValueError, match="weights or normalization buffers changed"):
            model.validate_prepared_state(prepared)

    model = _moment_core("M", thermal=False)
    model.train()
    prepared = model.prepare(scene)
    assert not prepared.graph_lifetime.consumed
    prepared.moment_hyperedges.processed_state.sum().backward()
    assert prepared.graph_lifetime.consumed
    with pytest.raises(ValueError, match="graph was consumed"):
        model.validate_prepared_state(prepared)


def test_moment_core_thermal_vjp_and_bounded_export_share_source_coefficients():
    model = _moment_core("M")
    model.eval()
    with torch.no_grad():
        model.hyperedge_processor.context_projection.weight.fill_(0.002)
        model.hyperedge_processor.heat_constant_projection.weight.fill_(0.001)
        model.hyperedge_processor.heat_spatial_projection.weight.fill_(0.001)
    scene = _scene_batch()
    coordinates = torch.tensor([[[0.18, 0.21], [0.82, 0.75]]])
    base_query = QueryBatch(
        coordinates=coordinates,
        features=torch.tensor([[[0.1], [-0.2]]]),
        query_mask=torch.ones(1, 2, dtype=torch.bool),
        query_ids=torch.tensor([[501, 502]]),
        scene_key=scene.scene_key,
        coordinate_frame_key=scene.coordinate_frame_key,
        receiver_key="moment-query-fixture",
    )
    from honf_forward_core.inherited_attention import smoothstep_grid_receiver_cover, with_receiver_cover

    cover = smoothstep_grid_receiver_cover(
        base_query.coordinates,
        scene.environment_coordinates,
        scene.environment_ids,
        scene.environment_mask,
        base_query.query_ids,
        frame_key=scene.coordinate_frame_key,
        query_mask=base_query.query_mask,
    )
    query = with_receiver_cover(base_query, cover)
    organization = model.prepare(scene)
    assert torch.count_nonzero(organization.moment_hyperedges.heat_mix_constant) > 0
    assert torch.count_nonzero(organization.moment_hyperedges.heat_mix_spatial) > 0
    package = organization.moment_hyperedges
    requested_dtype = torch.float64
    density = (organization.module_atlas.probabilities.to(requested_dtype)
               / organization.module_atlas.normalized_measure.to(requested_dtype)[:, None, None, :])
    relative = (package.module_coordinates.to(requested_dtype)[:, None, :, :]
                - package.environment_coordinates.to(requested_dtype)[:, :, None, :])
    spatial = torch.einsum(
        "bherd,bemd->bherm", package.heat_mix_spatial.to(requested_dtype), relative
    )
    literal_coefficients = (
        density[..., None] * organization.source_carriers.to(requested_dtype)[:, :, None, :, :]
    ).permute(0, 1, 2, 4, 3)
    literal_coefficients = literal_coefficients * (
        1.0 + package.heat_mix_constant.to(requested_dtype)[..., None] + spatial
    )
    torch.testing.assert_close(
        model._source_transport_coefficients(organization, requested_dtype),
        literal_coefficients,
        rtol=2e-6,
        atol=2e-7,
    )
    controls = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        counted = model.decode(organization, query, controls.detach(), accumulation_dtype=torch.float64)
    assert counted.work_counts["projected_context_gather_value_reads"] == (
        counted.work_counts["projected_context_gather_slots"] * 128
    )
    assert counted.work_counts["projected_context_gather_bytes"] == (
        counted.work_counts["projected_context_gather_value_reads"]
        * organization.prepared_readout_values.values.element_size()
    )
    predicted = model.apply_increment(organization, query, controls, accumulation_dtype=torch.float64)
    kernels = model.export_source_kernels(organization, query, accumulation_dtype=torch.float64)
    torch.testing.assert_close(predicted, (kernels * controls[:, None, :].to(torch.float64)).sum(dim=-1),
                               rtol=2e-5, atol=2e-6)
    cotangent = torch.tensor([[0.7, -0.4]])
    vjp = model.heat_vjp(organization, query, cotangent, accumulation_dtype=torch.float64)
    automatic = torch.autograd.grad((predicted * cotangent.to(predicted.dtype)).sum(), controls)[0]
    torch.testing.assert_close(vjp, automatic, rtol=3e-5, atol=3e-6)

    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    permuted_scene = replace(
        scene,
        module_features=scene.module_features.index_select(1, permutation),
        module_coordinates=scene.module_coordinates.index_select(1, permutation),
        module_mask=scene.module_mask.index_select(1, permutation),
        module_measure=scene.module_measure.index_select(1, permutation),
        module_ids=scene.module_ids.index_select(1, permutation),
        local_radius=scene.local_radius.index_select(1, permutation),
        module_characteristic_lengths=scene.module_characteristic_lengths.index_select(1, permutation),
    )
    permuted_organization = model.prepare(permuted_scene)
    permuted_controls = controls.detach().index_select(1, permutation)
    permuted_kernels = model.export_source_kernels(
        permuted_organization, query, accumulation_dtype=requested_dtype
    )
    torch.testing.assert_close(
        permuted_kernels.index_select(2, inverse), kernels, rtol=6e-5, atol=5e-6
    )
    permuted_prediction = model.apply_increment(
        permuted_organization, query, permuted_controls, accumulation_dtype=requested_dtype
    )
    torch.testing.assert_close(permuted_prediction, predicted, rtol=6e-5, atol=5e-6)


@pytest.mark.parametrize("variant", ["M", "H"])
def test_m_and_h_processors_have_active_gradients_after_two_updates(variant):
    processor = MomentHyperedgeProcessor(
        708, 2, processor=variant, thermal_transport=True, init_seed=73
    )
    generator = torch.Generator(device="cpu").manual_seed(641)
    descriptor = torch.randn(2, 5, 708, generator=generator)
    valid = torch.tensor([[1, 1, 1, 0, 1], [1, 0, 1, 1, 1]], dtype=torch.bool)
    context_probe = torch.randn(2, 5, 128, generator=generator)
    constant_probe = torch.randn(2, 4, 5, 8, generator=generator)
    spatial_probe = torch.randn(2, 4, 5, 8, 2, generator=generator)
    optimizer = torch.optim.AdamW(processor.parameters(), lr=1.0e-3, weight_decay=0.0)

    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = processor(descriptor, anchor_mask=valid)
        loss = ((output.context_residual * context_probe).sum()
                + (output.heat_mix_constant * constant_probe).sum()
                + (output.heat_mix_spatial * spatial_probe).sum())
        loss.backward()
        assert processor.context_projection.weight.grad is not None
        assert torch.count_nonzero(processor.context_projection.weight.grad) > 0
        assert processor.heat_constant_projection.weight.grad is not None
        assert torch.count_nonzero(processor.heat_constant_projection.weight.grad) > 0
        assert processor.heat_spatial_projection.weight.grad is not None
        assert torch.count_nonzero(processor.heat_spatial_projection.weight.grad) > 0
        assert processor.input_projection.weight.grad is not None
        if update == 0:
            assert torch.count_nonzero(processor.input_projection.weight.grad) == 0
        else:
            assert torch.count_nonzero(processor.input_projection.weight.grad) > 0
        optimizer.step()
