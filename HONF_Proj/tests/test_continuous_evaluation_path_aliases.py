"""Relocation compatibility accepts physical aliases without weakening identity."""
import copy
import hashlib

import pytest

from honf_runtime.continuous_joint_runtime import reconcile_evaluation_path_aliases


def identities(tmp_path):
    source = tmp_path / 'reference.npz'
    source.write_bytes(b'fixed TRAIN response evidence')
    alias = tmp_path / 'legacy-reference.npz'
    alias.symlink_to(source)
    row = dict(family_id='0001', path=str(source),
               sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    actual = dict(response_family_sources=[row], normalizer='fixed-training-scales')
    expected = copy.deepcopy(actual)
    expected['response_family_sources'][0]['path'] = str(alias)
    return actual, expected, source


def test_samefile_and_saved_hash_allow_path_alias_without_mutating_identity(tmp_path):
    actual, expected, _ = identities(tmp_path)
    before = copy.deepcopy(actual)
    normalized, aliases = reconcile_evaluation_path_aliases(actual, expected)
    assert normalized == expected and actual == before
    assert len(aliases) == 1 and aliases[0]['kind'] == 'response_family_source'
    assert aliases[0]['actual_path'] != aliases[0]['inherited_path']


def test_identical_duplicate_file_is_not_a_physical_alias(tmp_path):
    actual, expected, source = identities(tmp_path)
    duplicate = tmp_path / 'duplicate.npz'
    duplicate.write_bytes(source.read_bytes())
    actual['response_family_sources'][0]['path'] = str(duplicate)
    with pytest.raises(ValueError, match='same physical file'):
        reconcile_evaluation_path_aliases(actual, expected)


def test_samefile_with_changed_reference_bytes_is_rejected(tmp_path):
    actual, expected, source = identities(tmp_path)
    source.write_bytes(b'changed physical evidence')
    with pytest.raises(ValueError, match='saved hash'):
        reconcile_evaluation_path_aliases(actual, expected)


@pytest.mark.parametrize('field', ['family_id', 'normalizer'])
def test_physical_path_alias_cannot_hide_other_identity_changes(tmp_path, field):
    actual, expected, _ = identities(tmp_path)
    if field == 'family_id':
        actual['response_family_sources'][0]['family_id'] = 'other'
    else:
        actual['normalizer'] = 'refitted-scales'
    with pytest.raises(ValueError, match='identity differs|data/transforms'):
        reconcile_evaluation_path_aliases(actual, expected)


def test_native_cache_alias_requires_same_directory_and_exact_recipe(tmp_path):
    current = tmp_path / 'shared-catalogue'
    current.mkdir()
    alias = tmp_path / 'legacy-cache'
    alias.symlink_to(current, target_is_directory=True)
    actual = dict(native_role_catalogue_cache=dict(persistent_directory=str(current), recipe='exact'))
    expected = dict(native_role_catalogue_cache=dict(persistent_directory=str(alias), recipe='exact'))
    normalized, aliases = reconcile_evaluation_path_aliases(actual, expected)
    assert normalized == expected and aliases[0]['same_directory_inode']
    actual['native_role_catalogue_cache']['recipe'] = 'different'
    with pytest.raises(ValueError, match='data/transforms'):
        reconcile_evaluation_path_aliases(actual, expected)
