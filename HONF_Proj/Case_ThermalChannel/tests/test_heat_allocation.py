import numpy as np
import pytest
from channelthermal.heat_allocation import allocate_minimax_heat


def test_known_two_source_optimum_with_physical_column_order_and_caps():
    proposal=allocate_minimax_heat([[2.,0.],[0.,1.]],('source_b','source_a'),
        lower=0.,upper=[1.,3.],total_heat=3.)
    assert proposal.module_ids==('source_b','source_a')
    np.testing.assert_allclose(proposal.heat,[1.,2.],atol=1e-12)
    np.testing.assert_allclose(proposal.predicted_temperature,[2.,2.],atol=1e-12)
    assert proposal.physical_verification=='unverified'


def test_heat_cannot_repair_pressure_and_no_implicit_constraint_tolerance():
    proposal=allocate_minimax_heat([[1.,2.]],(11,12),lower=0.,upper=2.,total_heat=2.,
        layout_pressure=1.000001,pressure_limit=1.)
    assert not proposal.surrogate_feasible and proposal.heat is None
    assert proposal.status=='layout_pressure_infeasible_heat_independent'


def test_impossible_total_and_incomplete_constraints_are_rejected():
    with pytest.raises(ValueError,match='no feasible'):
        allocate_minimax_heat([[1.,2.]],(11,12),lower=1.,upper=2.,total_heat=1.)
    with pytest.raises(ValueError,match='together'):
        allocate_minimax_heat([[1.]],(11,),lower=0.,upper=2.,total_heat=1.,layout_pressure=1.)
