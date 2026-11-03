import h5py
import numpy as np
import pytest

from channelthermal.interaction_evidence.types import DesignState, EvidenceSplit, ModuleState, OperatingContext, RoleOutput, PhysicalSolveOutput, MeasuredQuantity, SolveRecord, SolveStatus, EvidenceSource
from channelthermal.inverse.joint_edit_reference import edited_design, load_eligible_atlas, native_reference_quantities, aligned_field_metrics


def _design():
    return DesignState('0001', 'family', EvidenceSplit.TRAIN, (
        ModuleState('a', (0.5, 0.5), 3), ModuleState('b', (1.5, 0.5), 4), ModuleState('pad', (0., 0.), 0, False)))


def test_eligibility_checks_original_source_before_atlas(tmp_path):
    path = tmp_path / 'data.h5'
    with h5py.File(path, 'w') as h:
        h.create_group('cases/0001').attrs['split'] = 'test'
    with pytest.raises(ValueError, match='source split'):
        load_eligible_atlas(path, '0001', ['0001'], tmp_path / 'absent.npz')
    with pytest.raises(ValueError, match='original TRAIN'):
        load_eligible_atlas(path, '0001', ['other'], tmp_path / 'absent.npz')


def test_edit_preserves_original_slots_and_controls():
    base = _design()
    xy = np.array([m.position_xy for m in base.modules])
    xy[0, 0] += .1
    result = edited_design(base, xy)
    assert result.modules[0].position_xy == pytest.approx((.6, .5))
    assert result.modules[1:] == base.modules[1:]
    assert result.modules[0].heating == base.modules[0].heating
    xy[2, 0] = 1
    with pytest.raises(ValueError, match='present'):
        edited_design(base, xy)


def _role(name, xy, values, names, valid, ids, receivers=None):
    return RoleOutput(name, np.asarray(xy), np.asarray(values), tuple(names), tuple('unit' for _ in names),
                      np.asarray(valid), np.ones(len(ids)), tuple(ids), receivers)


def test_native_objective_and_mask_alignment():
    design = _design()
    fluid = _role('fluid_fields', [[0, 0], [.5, .5], [2, 0]], [[4], [100], [1]], ['p'], [[1], [0], [1]], ['f0', 'f1', 'f2'])
    solid = _role('solid_temperature', [[0, 0], [0, 1], [0, 0]], [[1], [100], [3]], ['temperature'], [[1], [0], [1]], ['a0', 'a1', 'b0'], ('a', 'a', 'b'))
    interface = _role('interface', [[0, 0]], [[0]], ['T_surface'], [[1]], ['i0'], ('a',))
    output = PhysicalSolveOutput(roles={'fluid_fields': fluid, 'solid_temperature': solid, 'interface': interface}, quantities={'pressure_drop': MeasuredQuantity(3., 'unit')}, active_module_ids=('a','b'), module_peak_temperature={'a':1, 'b':3}, units_metadata={})
    record = SolveRecord(record_id='r', design=design, context=OperatingContext({'module_radius':.2, 'domain_length_x':2}), source=EvidenceSource.REFERENCE_SOLVER, status=SolveStatus.CONVERGED, elapsed_seconds=1., provenance={}, output=output)
    out = native_reference_quantities(record, 1.)
    assert out['objective'] == pytest.approx(3 + np.log((np.exp(-2)+1)/2))
    assert out['pressure_drop'] == 3 and out['material_count'] == 2
    metrics = aligned_field_metrics(solid, np.array([[2], [999], [5]]), query_ids=solid.query_ids, query_features=solid.query_features)
    assert metrics['temperature']['rmse'] == pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError, match='identities'):
        aligned_field_metrics(solid, solid.values, query_ids=solid.query_ids[::-1], query_features=solid.query_features)


def test_reference_reuse_rejects_solver_and_extraction_changes():
    from channelthermal.inverse.joint_edit_reference import compatible_reference
    design = _design()
    fluid = _role('fluid_fields', [[0, 0]], [[1]], ['p'], [[1]], ['f0'])
    solid = _role('solid_temperature', [[0, 0], [0, 0]], [[1], [2]], ['temperature'], [[1], [1]], ['a0', 'b0'], ('a', 'b'))
    interface = _role('interface', [[0, 0]], [[0]], ['T_surface'], [[1]], ['i0'], ('a',))
    roles = {'fluid_fields': fluid, 'solid_temperature': solid, 'interface': interface}
    output = PhysicalSolveOutput(roles=roles, quantities={'pressure_drop': MeasuredQuantity(0., 'unit')}, active_module_ids=('a', 'b'), module_peak_temperature={'a': 1, 'b': 2}, units_metadata={})
    context = OperatingContext({'module_radius': .2, 'domain_length_x': 2})
    record = SolveRecord(record_id='r', design=design, context=context, source=EvidenceSource.REFERENCE_SOLVER, status=SolveStatus.CONVERGED, elapsed_seconds=1., provenance={}, output=output)
    config = {'thermal': {'max_solve_time': 200, 'stop_on_convergence': True}, 'save': {'case_id': 'old'}}
    assert compatible_reference(record, design, context, config, {**config, 'save': {'case_id': 'new'}}, expected_roles=roles)
    assert not compatible_reference(record, design, context, config, {'thermal': {'max_solve_time': 50, 'stop_on_convergence': True}}, expected_roles=roles)
    changed = dict(roles)
    changed['solid_temperature'] = _role('solid_temperature', [[0, 0], [0, 0]], [[1], [2]], ['temperature'], [[1], [0]], ['a0', 'b0'], ('a', 'b'))
    assert not compatible_reference(record, design, context, config, config, expected_roles=changed)
