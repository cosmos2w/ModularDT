import copy

import numpy as np
import pytest

from channelthermal.inverse.response_search import (
    PhysicalBudget, PhysicalFieldCache, SearchBudget, accept_portfolio, correlation_covariance,
    physical_key, project_pressure, reduce_physical_fields, search, select_finalists, validate_physical_convergence,
)


def config():
    return {"domain": {"module_radius": 1., "nx": 3, "ny": 2, "lx": 6., "ly": 3.},
            "flow": {"re": 100., "dt": .01}, "thermal": {"k": 2.},
            "layout": {"centers": [[2., 1.], [4., 2.], [5., 1.]], "heat_powers": [1., 2., 3.]},
            "local_module": {"size": 4}}


def run(policy, *, valid=lambda _: True, ids=(10, 20, 30), budget=None):
    cfg = config()
    source = copy.deepcopy(cfg)
    def scorer(c):
        assert {**c, "layout": source["layout"]} == source
        d = np.asarray(c["layout"]["centers"]) - np.asarray(source["layout"]["centers"])
        return {"f": 10 + np.sum(d * [[2., -3.], [1., 4.], [-2., 1.]]),
                "p": 5 + np.sum(d * [[1., .5], [.2, -.3], [0., 1.]]), "query_pairs": 7}
    output = search(cfg, policy=policy, original_member_ids=ids, service=[.7, .2, .1],
        affinity=np.array([[1., .4, .1], [.4, 1., .2], [.1, .2, 1.]]),
        baseline={"f": 10., "p": 5.}, pressure_allowance=.02, roi_scale=.3,
        layout_index=1, request_index=0, valid=valid, score=scorer, budget=budget)
    assert cfg == source
    return output


def test_s_probes_centered_one_sided_and_48_charged_slots():
    budget = SearchBudget()
    # Disallow positive x of the first member; all invalid attempts remain charged.
    out = run("H-S", valid=lambda c: c["layout"]["centers"][0][0] <= 2., budget=budget)
    assert len(out["slots"]) == budget.slots == 48
    assert len(out["nominated"]) == 2
    assert sum(row["probe"] is not None for row in out["slots"]) == 8
    assert budget.scene_calls == out["actual_scene_calls"]
    assert budget.query_pairs == 7 * budget.scene_calls
    for member in out["nominated"]:
        expected_f = {10: [2., -3.], 20: [1., 4.], 30: [-2., 1.]}[member]
        np.testing.assert_allclose(out["gradients"][member]["f"], expected_f)
    if 10 in out["nominated"]:
        assert out["gradients"][10]["method"][0] == "backward"
    for row in out["slots"]:
        delta = row["centers"] - np.asarray(config()["layout"]["centers"])
        assert np.linalg.norm(delta, axis=1).max() <= .15 + 1e-12
    assert out["raw_slot"] >= 8


def test_common_uniform_draws_and_adaptive_batches():
    a, b, u = run("H-A"), run("G-A"), run("U")
    assert len(a["slots"]) == len(u["slots"]) == 48
    assert len(a["mean_history"]) == 4
    assert any(np.any(x != 0) for x in a["mean_history"])
    first = a["slots"][:12]
    ranked = sorted(first, key=lambda row: (
        -((10. - row["prediction"]["f"]) / .3 - max(0., (row["delta_p"] - .02) / .04)), row["slot"]))
    source = np.asarray(config()["layout"]["centers"])
    expected = .5 * np.mean([row["centers"] - source for row in ranked[:4]], axis=0)
    np.testing.assert_allclose(a["mean_history"][0], expected)
    for x, y in zip(a["slots"], b["slots"]):
        assert x["branch"] == y["branch"]
        assert x["block"] == y["block"]
        np.testing.assert_array_equal(x["centers"], y["centers"])
    assert all(x["branch"] == "uniform" for x in u["slots"])
    z = correlation_covariance([[4., 2., 0.], [2., 4., 0.], [0., 0., 0.]])
    np.testing.assert_allclose(z, [[1., .45, 0.], [.45, 1., 0.], [0., 0., 1.]])


def test_projection_is_joint_shrink_and_uses_linear_allowance():
    v = np.array([[.3, .05], [.1, .4]])
    g = np.array([[1., -1.], [2., 1.]])
    out = project_pressure(v, g, .01, .15)
    assert np.sum(g * out) <= .01 + 1e-12
    assert np.linalg.norm(out, axis=1).max() <= .15 + 1e-12
    projected = v - max(0., (np.sum(g*v)-.01)/np.sum(g*g))*g
    np.testing.assert_allclose(out, projected * min(1., .15 / np.linalg.norm(projected, axis=1).max()))


def test_finalists_selected_before_truth_and_incomplete_portfolio():
    def row(slot, f, p):
        return {"slot": slot, "valid": True, "baseline": False,
                "prediction": {"f": f, "p": p}, "delta_p": p - 5.}
    rows = [row(0, 8., 5.2), row(1, 9., 5.01), row(2, 9., 5.)]
    finalists = select_finalists(10., rows, .05)
    assert finalists == {"feasible": 1, "objective": 0, "portfolio": [1, 0]}
    values = {1: {"status": "solved", "f": 9.3, "p": 5.01},
              0: {"status": "solved", "f": 8.8, "p": 5.04}}
    outcome = accept_portfolio({"f": 10., "p": 5.}, finalists["portfolio"], values, .05, .5)
    assert outcome["status"] == "resolved" and outcome["accepted"] == 0 and outcome["above_resolution"]
    outcome = accept_portfolio({"f": 10., "p": 5.}, finalists["portfolio"],
                               {1: values[1], 0: {"status": "unresolved_timeout"}}, .05, .5)
    assert outcome["status"] == "incomplete" and outcome["gain"] is None
    assert outcome["provisional_accepted"] == 1


def test_complete_fields_are_reduced_per_request_and_policy_is_in_key():
    field = np.zeros((4, 5)); field[:, 4] = [1., 3., 10., 20.]; field[:, 2] = [5., 1., 4., 2.]
    xy = [[.1, 0.], [9.9, 0.], [.1, 1.], [9.9, 1.]]
    a = reduce_physical_fields(field, [1]*4, xy, [1, 1, 0, 0], 10.)
    b = reduce_physical_fields(field, [1]*4, xy, [0, 0, 1, 1], 10.)
    assert a == {"f": 2., "p": 3.} and b == {"f": 15., "p": 3.}
    assert physical_key(config(), {"tol": 1.e-4}) != physical_key(config(), {"tol": 1.e-5})
    fields = {"field": field, "fluid_mask": [1] * 4, "fluid_xy": xy, "domain_length_x": 10.}
    cache = PhysicalFieldCache(lambda reference: fields)
    key = cache.remember(config(), {"tol": 1.e-4}, "retained_full_field")
    assert cache.reduce(key, [1, 1, 0, 0]) == a
    assert cache.reduce(key, [0, 0, 1, 1]) == b
    other_key = cache.remember(config(), {"tol": 1.e-5}, "distinct_policy_field")
    assert other_key != key
    with pytest.raises(ValueError, match='different retained reference'):
        cache.remember(config(), {"tol": 1.e-4}, "wrong_field")
    with pytest.raises(ValueError, match="mask"):
        reduce_physical_fields(field, [1, 0, 1, 1], xy, [1, 1, 0, 0], 10.)


def test_caps_and_reject_fp32_source():
    with pytest.raises(ValueError): PhysicalBudget(max_starts=801)
    b = PhysicalBudget(max_starts=1)
    assert b.start('design') and not b.start('design')
    b.finish('design', {'status': 'failed'}, .1)
    with pytest.raises(RuntimeError): b.start('new')
    with pytest.raises(RuntimeError, match='scene-slot'):
        run('U', budget=SearchBudget(max_slots=2))
    cfg = config(); cfg['layout']['centers'] = np.asarray(cfg['layout']['centers'], dtype=np.float32)
    with pytest.raises(ValueError, match='FP64'):
        search(cfg, policy='U', original_member_ids=[0, 1, 2], service=[1, 1, 1], affinity=np.eye(3),
               baseline={'f': 1., 'p': 1.}, pressure_allowance=.1, roi_scale=1., layout_index=0,
               request_index=0, valid=lambda _: True, score=lambda _: {'f': 1., 'p': 1.})


def test_all_invalid_probes_and_proposals_do_not_spend_calls_or_add_slots():
    budget = SearchBudget()
    result = run('G-S', valid=lambda _: False, budget=budget)
    assert budget.slots == 48 and budget.scene_calls == 0
    assert all(not row['valid'] for row in result['slots'])
    assert all(np.isnan(result['gradients'][member]['f']).all() for member in result['nominated'])
    assert result['finalists']['portfolio'] == [] and result['raw_slot'] is None


def test_convergence_requires_the_full_saved_window_and_consistent_runtime():
    rows = [dict(converged=0, convergence_window_ok=0, time=1., heat_active=1,
                 delta_inf=1.e-5, delta_l2_rel=1.e-6, step=1),
            dict(converged=1, convergence_window_ok=1, time=2., heat_active=1,
                 delta_inf=2.e-5, delta_l2_rel=2.e-6, step=2)]
    cfg = {'thermal': {'convergence_window': 2, 'convergence_tol': 1.e-4,
                       'convergence_rel_tol': 1.e-5, 'stop_on_convergence': True,
                       'min_solve_time': 1., 'require_heat_active_for_convergence': True},
           'runtime': {'converged': True, 'converged_step': 2, 'converged_time': 2.,
                       'final_delta_inf': 2.e-5, 'final_delta_l2_rel': 2.e-6}}
    assert validate_physical_convergence(rows, cfg)['window'] == 2
    bad = copy.deepcopy(rows); bad[0]['delta_inf'] = 2.e-4
    with pytest.raises(ValueError, match='window exceeds'):
        validate_physical_convergence(bad, cfg)
    bad = copy.deepcopy(rows); bad[0]['delta_l2_rel'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        validate_physical_convergence(bad, cfg)
    bad = [copy.deepcopy(rows[0]), copy.deepcopy(rows[0]), copy.deepcopy(rows[1])]
    bad[1]['delta_inf'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        validate_physical_convergence(bad, cfg)
    bad = copy.deepcopy(cfg); bad['runtime']['converged_step'] = 3
    with pytest.raises(ValueError, match='metadata'):
        validate_physical_convergence(rows, bad)


def test_completed_score_is_journaled_before_a_post_call_budget_stop():
    class ExhaustedAfterCall(SearchBudget):
        def charge_gpu(self, seconds, *, scene_calls=0, query_pairs=0):
            super().charge_gpu(seconds, scene_calls=scene_calls, query_pairs=query_pairs)
            raise RuntimeError('time cap')

    saved = []
    with pytest.raises(RuntimeError, match='time cap'):
        search(config(), policy='U', original_member_ids=[10, 20, 30], service=[1, 1, 1],
               affinity=np.eye(3), baseline={'f': 1., 'p': 1.}, pressure_allowance=.1,
               roi_scale=1., layout_index=0, request_index=0, valid=lambda _: True,
               score=lambda _: {'f': .9, 'p': 1.}, budget=ExhaustedAfterCall(), observer=saved.append)
    assert len(saved) == 1 and saved[0]['slot'] == 0 and saved[0]['prediction']['f'] == .9
    assert 'time cap' in saved[0]['error']


def test_invalid_scorer_output_still_journals_the_charged_scene():
    saved = []
    with pytest.raises(KeyError):
        search(config(), policy='U', original_member_ids=[10, 20, 30], service=[1, 1, 1],
               affinity=np.eye(3), baseline={'f': 1., 'p': 1.}, pressure_allowance=.1,
               roi_scale=1., layout_index=0, request_index=0, valid=lambda _: True,
               score=lambda _: {'f': .9}, observer=saved.append)
    assert len(saved) == 1 and saved[0]['prediction'] == {'f': .9}
    assert saved[0]['delta_p'] is None and saved[0]['error']
