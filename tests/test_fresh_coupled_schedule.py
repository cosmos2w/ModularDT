"""The opt-in schedule follows the saved R17-4 absolute epoch clock."""
import math

import pytest

from channelthermal.training.thermal_native_pair_task import fresh_coupled_schedule
from honf_runtime.unified_training import ScheduleSpec


def test_historical_schedule_and_legacy_constant_have_distinct_late_rates():
    recipe = {'optimizer': {'schedule': 'hold1000_cosine2000', 'lr': 3e-4, 'final_lr': 3e-5},
              'training': {'optimizer_schedule_horizon': 2000, 'local_horizon': 1000}}
    scheduled = ScheduleSpec(**fresh_coupled_schedule(recipe))
    assert scheduled.value(1) == scheduled.value(500) == scheduled.value(1000) == 3e-4
    assert math.isclose(scheduled.value(1500), 1.65e-4, rel_tol=1e-14)
    assert math.isclose(scheduled.value(2000), 3e-5, rel_tol=1e-14)
    assert math.isclose(scheduled.value(2500), 3e-5, rel_tol=1e-14)
    recipe['optimizer']['schedule'] = 'constant_config_lr'
    legacy = ScheduleSpec(**fresh_coupled_schedule(recipe))
    assert legacy.value(1500) == legacy.value(2500) == 3e-4
    recipe['optimizer']['schedule'] = 'hold1000_cosine2000'
    recipe['training']['optimizer_schedule_horizon'] = 1500
    with pytest.raises(ValueError, match='2000-epoch'):
        fresh_coupled_schedule(recipe)
