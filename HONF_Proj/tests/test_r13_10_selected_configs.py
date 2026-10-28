"""The selected L qualification and formal A retain their declared contracts."""
from pathlib import Path
from honf_runtime.contextual_frontier_training import load_contextual_frontier_config, _anchored_schedule_and_model

CONFIGS = Path(__file__).resolve().parents[1] / 'configs/research'

def test_selected_configs_and_l20_schedule_are_explicit():
    l, _ = load_contextual_frontier_config(CONFIGS / 'thermal_r13_10l_local_e_cover_train150_v1.json')
    f, _ = load_contextual_frontier_config(CONFIGS / 'thermal_r13_10_e_coverage_train600_formal_v1.json')
    for cfg in (l, f):
        model = cfg['member_routing']['model']
        assert model['access_measure'] == 'inventory_unit_total'
        assert model['ee_incidence_aggregation'] == 'post_gate'
        assert model['score_calibration'] == 'inherited'
    assert (l['training']['local_horizon'], l['training']['first_segment_epochs'], l['training']['checkpoint_epochs']) == (20, 20, [20])
    assert (f['training']['local_horizon'], f['training']['first_segment_epochs'], f['training']['optimizer_schedule_horizon']) == (500, 100, 500)
    assert f['training']['membership'] == 'full_train600'
    assert f['initialization']['normalization_mode'] == 'scratch_train600_stats'
    allowed, _ = _anchored_schedule_and_model(l['training'], l['initialization'], l['member_routing']['model'], 'R13-8-A-EAnchor')
    assert not allowed
    historical, _ = load_contextual_frontier_config(CONFIGS / 'thermal_r13_9a_e_anchor_train600_formal_v1.json')
    assert 'access_measure' not in historical['member_routing']['model']
    assert 'ee_incidence_aggregation' not in historical['member_routing']['model']


def test_u_local_cover_paired_union_and_live_cover_gradient():
    import torch
    from tests.test_e_anchored_packed_execution import fixture
    model, scene, query = fixture('local_e_cover')
    model.applied_incidence.ee_incidence_aggregation = 'post_gate'
    prepared = model.prepare(scene)
    full, restricted, work = model.decode_paired(prepared, query)
    standalone = model.decode(prepared, query, intervention={'read_mode': 'full'})
    for role in ('M', 'E'):
        key = 'query_unique_union_raw_score_pairs_' + role
        assert work[key] == standalone.work_counts[key]
    grads = torch.autograd.grad((full.joint_fields + restricted.joint_fields).square().sum(),
                                (query.coordinates, query.anchor_weights))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
