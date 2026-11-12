"""The formal S access setting is separate from the ongoing X screen."""
import json
from pathlib import Path

import pytest

from honf_forward_core.interaction_reused_member_routing.detail import DetailAnchoredMemberRoutingCore
from channelthermal.training.r16_8_content_task import enable_content


@pytest.mark.parametrize('arm', ('b0', 'e', 'r', 'd', 'v'))
def test_formal_s_and_screen_x_access_survive_constructor_roundtrip(arm):
    configs = Path('configs/research')
    formal = json.loads((configs / f'thermal_r16_10_{arm}_train600.json').read_text())
    screen = json.loads((configs / f'thermal_r16_10_{arm}_train150.json').read_text())
    for recipe, expected in ((formal, 'inventory_unit_total'), (screen, 'active_domain')):
        core_config = recipe['member_routing']['model']
        assert core_config['access_measure'] == expected
        core = DetailAnchoredMemberRoutingCore.from_config(core_config)
        assert core.access_measure == expected
        saved = core.get_config()
        assert DetailAnchoredMemberRoutingCore.from_config(saved).access_measure == expected
        if expected == 'inventory_unit_total':
            assert 'access_measure' not in saved  # historical default serialization
        if arm == 'v':
            enable_content(core, 'C8')
            assert core.access_measure == expected
