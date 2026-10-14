"""Scientific contracts for the explicitly selected typed-centroid reader."""

from dataclasses import fields, replace

import pytest
import torch

from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from test_content_hypergraph_v2_core import _core, _query, _scene


MODE = "typed_w_centroids_v1"


def _double(value):
    return replace(value, **{
        item.name: getattr(value, item.name).double()
        for item in fields(value)
        if torch.is_tensor(getattr(value, item.name)) and getattr(value, item.name).is_floating_point()
    })


def test_revision_is_explicit_h6_only_and_roundtrips_without_changing_default_state():
    default = _core()
    explicit_default = _core(query_edge_geometry_mode=None)
    assert default.get_config() == explicit_default.get_config()
    assert "query_edge_geometry_mode" not in default.get_config()
    assert all(torch.equal(value, explicit_default.state_dict()[name])
               for name, value in default.state_dict().items())
    revised = _core(query_edge_geometry_mode=MODE)
    assert revised.get_config()["query_edge_geometry_mode"] == MODE
    rebuilt = ContentHypergraphCoreV2.from_config(revised.get_config())
    rebuilt.load_state_dict(revised.state_dict(), strict=True)
    assert rebuilt.get_config() == revised.get_config()
    common = set(default.state_dict()) & set(revised.state_dict())
    for name in common:
        if default.state_dict()[name].shape == revised.state_dict()[name].shape:
            assert torch.equal(default.state_dict()[name], revised.state_dict()[name]), name
    with pytest.raises(ValueError, match="A2-H6"):
        _core("A2-D", query_edge_geometry_mode=MODE)
    with pytest.raises(ValueError, match="query_edge_geometry_mode"):
        _core(query_edge_geometry_mode="unknown")


def test_centroids_use_final_w_once_and_remain_live_to_geometry_and_gather():
    core = _core(query_edge_geometry_mode=MODE).double()
    scene = _double(_scene())
    scene.module_coordinates.requires_grad_()
    scene.environment_coordinates.requires_grad_()
    org = core.prepare(scene)
    for typ in ("module", "environment"):
        weight = getattr(org, f"{typ}_gather_weights")
        coordinate = getattr(scene, f"{typ}_coordinates")
        center = getattr(org, f"{typ}_content_centroids")
        # Independent scalar accumulation: W already contains the measure.
        expected = torch.stack([
            torch.stack([sum(weight[b, h, j] * coordinate[b, j]
                             for j in range(coordinate.shape[1])) for h in range(6)])
            for b in range(scene.batch_size)
        ])
        torch.testing.assert_close(center, expected, atol=1e-12, rtol=1e-12)
        grad = torch.autograd.grad(center.square().sum(), (coordinate, weight), retain_graph=True)
        assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in grad)
    assert org.work_counts["content_centroid_coordinate_products"] == 2 * 6 * 7 * 2


@pytest.mark.parametrize("typ", ["module", "environment"])
def test_empty_type_has_zero_centroid_and_zero_reader_channels(typ):
    core = _core(query_edge_geometry_mode=MODE)
    scene = _scene()
    scene = replace(scene, **{f"{typ}_measure": torch.zeros_like(getattr(scene, f"{typ}_measure"))})
    org = core.prepare(scene)
    center = getattr(org, f"{typ}_content_centroids")
    valid = getattr(org, f"{typ}_content_valid")
    assert not valid.any()
    assert torch.equal(center, torch.zeros_like(center))
    output = core.decode(org, _query(), controls=torch.ones(2, 3))
    assert torch.isfinite(output.nonlinear_field).all()
    assert torch.isfinite(output.thermal_temperature).all()
    loss = output.nonlinear_field.square().sum() + output.thermal_temperature.square().sum()
    loss.backward()
    gate = getattr(core.edge_reader, f"{typ}_centroid_gate")
    assert gate.grad is not None and torch.equal(gate.grad, torch.zeros_like(gate.grad))


@pytest.mark.parametrize("name", ["module_content_centroids", "environment_content_centroids",
                                  "module_content_valid", "environment_content_valid"])
def test_public_prepared_boundary_rejects_modified_centroid_descriptors(name):
    core = _core(query_edge_geometry_mode=MODE)
    org = core.prepare(_scene())
    value = getattr(org, name)
    if value.dtype == torch.bool:
        value[0, 0].logical_not_()
    else:
        value[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="[Cc]entroid|content geometry"):
        core.decode(org, _query())


def test_centroid_reader_keeps_affine_heat_adjoint_tile_and_single_message_contracts():
    core = _core(query_edge_geometry_mode=MODE).double()
    scene, query = _double(_scene()), _double(_query())
    org = core.prepare(scene)
    controls = torch.tensor([[0.3, 0.8, 0.2], [0.1, 0.7, 0.9]], dtype=torch.float64, requires_grad=True)
    output = core.decode(org, query, controls=controls)
    zero = core.decode(org, query, controls=torch.zeros_like(controls))
    assert torch.equal(zero.thermal_temperature, torch.zeros_like(zero.thermal_temperature))
    assert torch.equal(zero.nonlinear_field, output.nonlinear_field)
    assert torch.autograd.grad(output.nonlinear_field.sum(), controls, allow_unused=True, retain_graph=True)[0] is None
    kernel = core.export_source_kernels(org, query)
    torch.testing.assert_close(output.thermal_temperature, torch.einsum("bqm,bm->bq", kernel, controls),
                               atol=1e-9, rtol=1e-7)
    tangent = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 7
    cotangent = torch.arange(10, dtype=torch.float64).reshape(2, 5) / 11
    left = (core.apply_increment(org, query, tangent) * cotangent).sum()
    right = (tangent * torch.autograd.grad((output.thermal_temperature * cotangent).sum(), controls)[0]).sum()
    torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-7)
    head_query = replace(query, coordinates=query.coordinates[:, :2], features=query.features[:, :2],
                         query_ids=query.query_ids[:, :2])
    head = core.decode(org, head_query, controls=controls)
    torch.testing.assert_close(head.nonlinear_field, output.nonlinear_field[:, :2], atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(head.thermal_temperature, output.thermal_temperature[:, :2], atol=1e-9, rtol=1e-7)
    assert output.work_counts["edge_message_rows"] == 2 * 5 * 6
    assert output.work_counts["edge_feature_message_entries"] == 2 * 5 * 6 * 32


def test_centroid_geometry_gradient_matches_rebuilt_smooth_scene_finite_difference():
    core = _core(query_edge_geometry_mode=MODE).double()
    scene, query = _double(_scene()), _double(_query())
    scene.module_coordinates.requires_grad_()

    def objective(value):
        org = core.prepare(value)
        result = core.decode(org, query, controls=torch.ones(2, 3, dtype=torch.float64),
                             local_gates=torch.zeros(2, 3, dtype=torch.float64))
        return result.nonlinear_field.square().mean() + result.thermal_temperature.square().mean()

    analytical = torch.autograd.grad(objective(scene), scene.module_coordinates)[0][0, 0, 0]
    step = 1e-6
    plus = scene.module_coordinates.detach().clone()
    minus = plus.clone()
    plus[0, 0, 0] += step
    minus[0, 0, 0] -= step
    measured = (objective(replace(scene, module_coordinates=plus))
                - objective(replace(scene, module_coordinates=minus))) / (2 * step)
    torch.testing.assert_close(analytical, measured, atol=1e-7, rtol=1e-5)


def test_centroid_revision_preserves_source_column_correspondence_under_permutation():
    core = _core(query_edge_geometry_mode=MODE).double()
    scene, query = _double(_scene()), _double(_query())
    permutation = torch.tensor([2, 0, 1])
    module_fields = ("module_features", "module_coordinates", "module_mask", "module_measure",
                     "module_ids", "local_radius", "module_characteristic_lengths")
    moved = replace(scene, **{name: getattr(scene, name)[:, permutation] for name in module_fields})
    org, moved_org = core.prepare(scene), core.prepare(moved)
    original_kernel = core.export_source_kernels(org, query)
    moved_kernel = core.export_source_kernels(moved_org, query)
    torch.testing.assert_close(moved_kernel, original_kernel[:, :, permutation], atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(moved_org.module_content_centroids, org.module_content_centroids,
                               atol=1e-9, rtol=1e-7)
    controls = torch.tensor([[0.3, 0.8, 0.2], [0.1, 0.7, 0.9]], dtype=torch.float64)
    output = core.decode(org, query, controls=controls)
    moved_output = core.decode(moved_org, query, controls=controls[:, permutation])
    torch.testing.assert_close(moved_output.thermal_temperature, output.thermal_temperature,
                               atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(moved_output.nonlinear_field, output.nonlinear_field,
                               atol=1e-9, rtol=1e-7)
