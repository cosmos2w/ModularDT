import copy
import random

import numpy as np
import pytest
import torch

from channelthermal.training.r16_10_diagnostics import calibrate_components, counterfactual_updates


def test_calibration_coefficients_groups_and_state():
    model = torch.nn.Linear(2, 1, bias=False)
    p = model.weight
    p.grad = torch.ones_like(p)
    saved = p.detach().clone()
    rng = torch.get_rng_state().clone()
    losses = {'native': lambda: (p * torch.tensor([[3., 4.]])).sum(),
              'endpoint': lambda: p.sum(),
              'response': lambda: (p * torch.tensor([[1., -1.]])).sum(),
              'decision': lambda: p[0, 0]}
    out = calibrate_components(model, losses, {'reader': ['weight']})
    assert out['coefficients']['endpoint'] == pytest.approx(.25 * 5 / 2**.5)
    assert out['coefficients']['response'] == pytest.approx(.10 * 5 / 2**.5)
    assert out['coefficients']['decision'] == pytest.approx(.25)
    assert out['groups']['reader']['cosines']['endpoint:response'] == pytest.approx(0.)
    assert torch.equal(p, saved) and torch.equal(p.grad, torch.ones_like(p))
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match='disconnected'):
        calibrate_components(model, {**losses, 'decision': lambda: torch.tensor(0.)})


def test_counterfactual_uses_adamw_and_restores_run_state():
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.01)
    p = model.weight
    (p.square().sum()).backward(); opt.step(); opt.zero_grad()
    state = copy.deepcopy(opt.state_dict())
    weight = p.detach().clone()
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    py_state, np_state, rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    losses = {'native': lambda: (p - 1).square().sum(),
              'endpoint': lambda: (p + 1).square().sum(),
              'response': lambda: (2*p - 1).square().sum()}
    result = counterfactual_updates(model, opt, losses, {'native': 1., 'endpoint': .3, 'response': .1}, .2)
    assert result['updates']['native_only']['displacement_norm'] > 0
    assert result['updates']['combined']['clip_scale'] < 1
    assert torch.equal(p, weight) and torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == py_state and np.array_equal(np.random.get_state()[1], np_state[1])
    assert opt.state_dict()['param_groups'] == state['param_groups']
    for key in state['state'][0]:
        assert torch.equal(opt.state_dict()['state'][0][key], state['state'][0][key])


def test_satisfied_hinge_is_valid_for_update_diagnostics_but_not_calibration():
    model = torch.nn.Linear(1, 1, bias=False)
    p = model.weight
    losses = {'native': lambda: (p - 2).square().sum(),
              'endpoint': lambda: (p + 1).square().sum(),
              'response': lambda: (p - 3).square().sum(),
              'decision': lambda: torch.relu(p - p.detach() - 1).square().sum()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    result = counterfactual_updates(model, optimizer, losses, {'native': 1., 'decision': .05})
    assert result['gradient_diagnostics']['groups']['global']['norms']['decision'] == 0
    with pytest.raises(ValueError, match='Zero gradient for decision'):
        calibrate_components(model, losses)


def test_identical_counterfactuals_start_from_identical_nonzero_adamw_moments():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    p = model.weight
    for target in (1., 3., -2.):
        optimizer.zero_grad(set_to_none=True)
        (p - target).square().sum().backward()
        optimizer.step()
    result = counterfactual_updates(model, optimizer,
        {'native': lambda: (p - 5).square().sum()}, {'native': 1.})
    assert result['updates']['native_only'] == result['updates']['combined']


def test_scratch_curves_record_calibrated_family_weights_and_sampled_validation(tmp_path):
    import json
    from honf_runtime.unified_training import _render_loss_curves

    coefficients = {'endpoint': .2863, 'response': .2574, 'decision': .00636}
    identity = {'recipe': {'r16_10_family': {'arm': 'R'}},
                'r16_10_family': {'coefficients': coefficients}}
    history = [{'epoch': 1, 'phase': 'joint', 'arm': 'R14-Detail-LocalECover',
                'case_visits': 150, 'train_losses': {'flow/u': .2,
                    'family/endpoint': .1, 'family/response': .05}}]
    _render_loss_curves(history, tmp_path, 'field_score', provider_identity=identity)
    metadata = json.loads((tmp_path / 'loss_curves_metadata.json').read_text())
    assert metadata['response_objective']['component_coefficients'] == coefficients
    assert 'VAL9/89' in metadata['validation_scope']
    assert 'excludes family' in metadata['response_objective']['validation_total']


def test_native_sampling_and_exact_matched_views(monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import thermal_r16_10_evaluate as ev
    ids = tuple(f'{i:04}' for i in range(150))
    chosen = ev.sampled_ids(ids)
    assert len(chosen) == 15 and chosen == ev.sampled_ids(ids)
    assert tuple(i for i in ids if i in chosen) == chosen
    calls = []
    monkeypatch.setattr(ev, 'evaluate_native_panel', lambda *a, **kw: calls.append((kw['mode'], kw['case_ids'])) or {})
    provider = type('Provider', (), {'training_case_ids': ids})()
    ev.evaluate_sampled_native(None, provider, epoch=1, split='train', case_ids=chosen, output=tmp_path / 'x')
    assert calls == [('restricted', chosen), ('full', chosen)]


def test_streamed_pieces_match_scalar_calibration_and_update():
    model = torch.nn.Linear(1, 1, bias=False)
    p = model.weight
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    scalar = {'native': lambda: (p - 2).square().sum(),
              'endpoint': lambda: (p + 1).square().sum(),
              'response': lambda: (p - 3).square().sum(),
              'decision': lambda: (p + 2).square().sum()}
    streamed = {key: (lambda fn=fn: (fn() / 2 for _ in range(2))) for key, fn in scalar.items()}
    a, b = calibrate_components(model, scalar), calibrate_components(model, streamed)
    assert a['coefficients'] == pytest.approx(b['coefficients'])
    weights = {'native': 1., 'endpoint': .25}
    a = counterfactual_updates(model, opt, scalar, weights)
    b = counterfactual_updates(model, opt, streamed, weights)
    assert a['updates']['combined']['displacement_norm'] == pytest.approx(b['updates']['combined']['displacement_norm'])


def test_family_predictions_are_saved_once_per_endpoint():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    from thermal_r16_10_evaluate import cache_family_predictions
    events = []
    def predict(family, endpoint, mode):
        events.append(('predict', family, endpoint, mode))
        return endpoint
    def save(key, prediction):
        events.append(('save', *key))
        return prediction
    result = cache_family_predictions(['0001'], lambda _: ['a', 'b'], predict, save)
    assert len(result) == 6
    assert [event[0] for event in events] == ['predict', 'save'] * 6
    assert list(result).count(('0001', 'baseline', 'restricted')) == 1


def test_decision_physical_violation_falls_back_without_reranking():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    from thermal_r16_10_evaluate import _research_decisions
    predicted = {'baseline': {'J': 0., 'P': 0.}, 'a': {'J': -2., 'P': 0.}, 'b': {'J': -1., 'P': 0.}}
    physical = {'baseline': {'J': 0., 'P': 0.}, 'a': {'J': -3., 'P': 2.}, 'b': {'J': -1., 'P': 0.}}
    rows = _research_decisions(predicted, physical, list(predicted), {'J': .01, 'P': .01},
                               {'J': 1., 'P': 1.}, {'J': 1., 'P': 1.})
    row = next(x for x in rows if x['objective'] == 'J' and x['multiplier'] == .5)
    assert row['model_selected_action'] == 'a' and row['accepted_action'] == 'baseline'
    assert row['raw_constraint_violation'] and row['physical_best_action'] == 'b'
    assert row['regret'] == pytest.approx(1.)
    assert len(rows) == 10
