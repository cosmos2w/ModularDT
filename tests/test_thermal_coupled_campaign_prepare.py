"""Matched recipe preparation preserves the cohort's scientific inputs."""
import copy

import pytest

from tools.thermal_coupled_campaign_prepare import prepare_recipes
from tools.verify_coupled_native_recovery import _sampled_reader_recipe


def source(cohort):
    size = 47 if cohort == 'train150' else 240
    calibration = 15 if cohort == 'train150' else 60
    return {'name': 'historical', 'training_profile': 'observation_response',
        'initialization': {'source_checkpoint': None, 'pretrained_weights': False, 'source_parent_age': 0},
        'training': {'membership': 'fixed25_train150_dev22' if cohort == 'train150' else 'full_train600',
                     'effective_batch_size': 48, 'microbatch_cases': 24, 'primary_queries': 1024},
        'optimizer': {'schedule': 'constant_config_lr', 'lr': .0003, 'final_lr': .00003},
        'member_routing': {'model': {'query_tile_size': 128}}, 'execution': {'precision': 'fp32'},
        'storage': {}, 'observation_response': {'mode': 'coupled_native', 'variant': 'sampled_reader_response',
            'pairs': [[str(2 * i), str(2 * i + 1)] for i in range(size)],
            'calibration_case_ids': [str(i) for i in range(calibration)], 'reader_seed': 170011,
            'anchor_count': 128}}


SCHEDULE = {'optimizer': {'schedule': 'hold1000_cosine2000', 'lr': 3e-4, 'final_lr': 3e-5},
            'training': {'optimizer_schedule_horizon': 2000}}


@pytest.mark.parametrize(('cohort', 'count', 'epochs'), [('train150', 3, 1000), ('train600', 2, 500)])
def test_recipe_cohort_preserves_data_and_only_changes_update_scope(cohort, count, epochs, tmp_path):
    template = source(cohort)
    before = copy.deepcopy(template)
    recipes = prepare_recipes(template, SCHEDULE, cohort=cohort, output_root=tmp_path / 'runs',
        initial_state=tmp_path / 'common.pt', round_name='R17_12', max_output_bytes=1000,
        peak_temporary_bytes=100, minimum_free_bytes=200)
    assert template == before and len(recipes) == count
    signatures = []
    scopes = set()
    for name, recipe in recipes.items():
        assert recipe['name'] == name
        assert recipe['training']['membership'] == template['training']['membership']
        assert recipe['training']['local_horizon'] == epochs
        assert recipe['training']['optimizer_schedule_horizon'] == 2000
        assert recipe['training']['checkpoint_epochs'] == list(range(100, epochs + 1, 100))
        assert recipe['training']['primary_queries'] == 1024
        assert recipe['member_routing']['model']['query_tile_size'] == 512
        options = recipe['observation_response']
        assert options['pairs'] == template['observation_response']['pairs']
        assert options['calibration_case_ids'] == template['observation_response']['calibration_case_ids']
        assert options['reader_policy'] == 'sampled_equal_probability'
        assert options['full_reader_probability'] == .5
        scopes.add(options['response_update_scope'])
        if options['response_update_scope'] == 'disabled':
            assert options['variant'] == 'native_control'
            assert 'response_gradient_composition' not in options
        else:
            assert options['variant'] == 'observed_response'
            assert options['response_gradient_composition']['maximum_native_norm_ratio'] == .1
        signature = copy.deepcopy(recipe)
        signature.pop('name')
        for key in ('variant', 'response_update_scope', 'response_gradient_composition'):
            signature['observation_response'].pop(key, None)
        signatures.append(signature)
    assert all(signature == signatures[0] for signature in signatures)
    assert scopes == ({'disabled', 'all_parameters', 'organization_parameters'} if cohort == 'train150'
                      else {'disabled', 'organization_parameters'})


def test_reject_wrong_cohort_or_nonhistorical_schedule(tmp_path):
    options = dict(output_root=tmp_path, initial_state=tmp_path / 'initial.pt', round_name='R17_12',
                   max_output_bytes=1000, peak_temporary_bytes=100, minimum_free_bytes=200)
    with pytest.raises(ValueError, match='cohort recipe'):
        prepare_recipes(source('train150'), SCHEDULE, cohort='train600', **options)
    changed = copy.deepcopy(SCHEDULE)
    changed['optimizer']['schedule'] = 'constant_config_lr'
    with pytest.raises(ValueError, match='R17-4 schedule'):
        prepare_recipes(source('train150'), changed, cohort='train150', **options)


def test_recovery_recognizes_legacy_and_explicit_reader_policy():
    assert _sampled_reader_recipe({'mode': 'coupled_native', 'variant': 'sampled_reader_response'})
    assert _sampled_reader_recipe({'mode': 'coupled_native', 'variant': 'observed_response',
                                   'reader_policy': 'sampled_equal_probability'})
    assert not _sampled_reader_recipe({'mode': 'coupled_native', 'variant': 'observed_response'})
