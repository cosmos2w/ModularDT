import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from thermal_dependency_flow_fit import flow_refinement_learning_rate


def test_near_boundary_refinement_schedule_hits_declared_ages():
    assert flow_refinement_learning_rate(1) == 1e-6
    assert flow_refinement_learning_rate(20) == 3e-5
    assert flow_refinement_learning_rate(21) == 3e-5
    assert flow_refinement_learning_rate(200) == 3e-5
    assert 3e-6 < flow_refinement_learning_rate(201) < 3e-5
    assert flow_refinement_learning_rate(500) == 3e-6


def test_near_boundary_refinement_schedule_rejects_out_of_range_ages():
    for age in (0, 501):
        try:
            flow_refinement_learning_rate(age)
        except ValueError:
            continue
        raise AssertionError(f"Unexpected schedule learning rate for age {age}.")
