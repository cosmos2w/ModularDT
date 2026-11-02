from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from channelthermal.inverse.relation_anchored import ThermalDesignEvaluator


class NativeFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.register_buffer('internal_temperature_std', torch.tensor([2.0]))
        self.structure = None

    def prepare_native(self, structure, fluid, *, local_query_points, ntheta):
        self.structure = structure
        assert ntheta == 64
        centers = structure['module_centers']
        valid_material = torch.ones(centers.shape[0], centers.shape[1] * local_query_points.shape[-2], dtype=torch.bool)
        return NS(structure=structure, scene_batch=NS(module_mask=structure['module_present'].bool()),
                  stencils={'material': NS(valid=valid_material), 'fluid': NS(valid=torch.ones(fluid.shape[:2], dtype=torch.bool))})

    def apply_native_with_work_counts(self, prepared, heat, *, intervention):
        assert intervention['read_mode'] in ('full', 'restricted')
        centers = prepared.structure['module_centers']
        material = (centers[..., 0] + heat)[..., None, None].expand(-1, -1, 2, -1)
        pressure = torch.stack((centers.sum(), centers.sum() * 2))[None, :, None]
        field = torch.cat((torch.zeros_like(pressure).expand(-1, -1, 2), pressure,
                           torch.zeros_like(pressure).expand(-1, -1, 2)), -1)
        return {'pred_internal_temperature': material, 'pred_field': field}, {'queries': 2}, None


def record():
    ids = ('a', 'b')
    modules = tuple(NS(module_id=key, position_xy=(3.0 + i * 3, 3.0), active=True, heating=1.0) for i, key in enumerate(ids))
    material = NS(query_features=np.tile([[0.0, 0.0], [0.1, 0.0]], (2, 1)), receiver_module_ids=('a', 'a', 'b', 'b'))
    interface = NS(receiver_module_ids=('a',) * 64 + ('b',) * 64)
    fluid = NS(query_features=np.array([[0.5, 3.0], [11.5, 3.0]]))
    context = dict(nu=.1, solid_alpha=.1, fluid_alpha=.1, solid_k=1., fluid_k=1., module_radius=.45,
                   domain_length_x=12., domain_length_y=6., re=50., u_in=1.)
    return NS(design=NS(modules=modules), context=NS(values=context),
              output=NS(roles={'fluid_fields': fluid, 'interface': interface, 'solid_temperature': material}))


def test_live_centers_heat_and_native_objectives_remain_differentiable():
    model = NativeFixture()
    evaluator = ThermalDesignEvaluator(model, record())
    x = torch.tensor([[3., 3.], [6., 3.]], requires_grad=True)
    heat = torch.tensor([1., 1.], requires_grad=True)
    result = evaluator.evaluate_design(x, heat=heat, read_mode='restricted')
    dx, dh = torch.autograd.grad(result.objective + result.pressure_drop, (x, heat))
    assert model.structure['module_source_ids'] == ('a', 'b')
    assert result.valid_material_count == 4
    assert result.pressure_drop.item() == pytest.approx(-15.)
    assert evaluator.tau == pytest.approx(.1)
    assert torch.isfinite(dx).all() and torch.isfinite(dh).all()
    assert dh.sum().item() == pytest.approx(1., abs=2e-6)
    assert not model.weight.requires_grad


def test_rejects_invalid_read_mode():
    evaluator = ThermalDesignEvaluator(NativeFixture(), record())
    with pytest.raises(ValueError, match='read mode'):
        evaluator.evaluate_design(torch.tensor([[3., 3.], [6., 3.]]), read_mode='implicit')


def test_r14_native_context_intervention_rebuilds_with_original_ids():
    from channelthermal.member_routing import ThermalDetailAnchoredMemberRoutingAdapter
    from tests.test_r14_runtime_integration import _config, STATS
    from tests.test_thermal_content_hypergraph_v2 import _structure, _receivers

    model = ThermalDetailAnchoredMemberRoutingAdapter(
        STATS, core_config=_config()['member_routing']['model'], nx=8, ny=6).eval()
    structure = {**_structure(), 'heat_powers': torch.ones(1, 2)}
    fluid, local = _receivers()
    normal = model.prepare_native(structure, fluid, local_query_points=local, ntheta=16)
    changed = model.prepare_native(
        structure, fluid, local_query_points=local, ntheta=16,
        context_message_removal={'stage': 'cycle0/me', 'route': 'me', 'destination_id': 31, 'donor_id': 0})
    assert not torch.equal(normal.organization.module_states, changed.organization.module_states)
