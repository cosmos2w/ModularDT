"""The new profile displays actual weighted totals and TRAIN-only calibration."""
import json

import numpy as np
from test_unified_loss_curves import _capture_figures

from honf_runtime.unified_training import _render_loss_curves


def test_profile_curves_preserve_pending_validation_and_calibrated_objective(tmp_path, monkeypatch):
    identity = {'state_response': {'enabled': True, 'warmup_epochs': 10, 'ramp_epochs': 40},
                'validation_scope': 'input_only_sample_of_declared_validation',
                'state_response_calibration': None}
    metadata = {'family/endpoint': {'panel': 'physics', 'label': 'FIT family endpoint'},
                'family/response': {'panel': 'physics', 'label': 'FIT family response'}}
    first = {'epoch': 1, 'phase': 'joint', 'arm': 'R14-Detail-LocalECover', 'case_visits': 5,
             'train_losses': {'flow/u': 2., 'family/endpoint': 0., 'family/response': 0.}}
    captured = _capture_figures(monkeypatch, tmp_path)
    _render_loss_curves([first], tmp_path, 'field_score', loss_metadata=metadata, provider_identity=identity)
    assert any('first review pending' in text.lower() for panel in captured[0]['texts'] for text in panel)
    saved = json.loads((tmp_path / 'loss_curves_metadata.json').read_text())
    assert saved['response_objective']['calibration'] == 'pending'
    assert 'no family labels enter native validation' in saved['panel_definitions']['physics'].lower()
    assert saved['validation_scope'] == identity['validation_scope']

    calibration = {'source': 'TRAIN panel', 'coefficients': {'endpoint': .3, 'response': .2}}
    identity['state_response_calibration'] = calibration
    second = {'epoch': 20, 'phase': 'joint', 'arm': 'R14-Detail-LocalECover', 'case_visits': 5,
              'train_losses': {'flow/u': 1., 'family/endpoint': .2, 'family/response': .1},
              'validation': {'field_score': .5},
              'validation_objective': {'terms': {'flow/u': .4}}}
    captured.clear()
    _render_loss_curves([first, second], tmp_path, 'field_score', loss_metadata=metadata, provider_identity=identity)
    assert captured[0]['titles'][0] == 'Optimized TRAIN and native validation totals'
    assert np.allclose(captured[0]['lines'][0][0], [2., 1.3])
    assert any('FIT family response' in label for panel in captured[0]['labels'] for label in panel)
    saved = json.loads((tmp_path / 'loss_curves_metadata.json').read_text())
    assert saved['response_objective']['calibration'] == calibration
    assert saved['response_objective']['component_coefficients'] == calibration['coefficients']
    assert saved['response_objective']['ramp'] == {'warmup_epochs': 10, 'ramp_epochs': 40}
