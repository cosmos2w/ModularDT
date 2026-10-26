from pathlib import Path
import h5py
import pytest
from hetmf.resources import link_shared_bundle


def _bundle(tmp_path):
    bundle = tmp_path / 'public'
    bundle.mkdir()
    for name in ('standard', 'expanded'):
        with h5py.File(bundle / f'{name}.h5', 'w') as h:
            h.attrs.update(schema='hetmf_training_h5_v1', accepted=True)
    for name in ('readiness', 'formal_standard', 'formal_expanded', 'pilot'):
        (bundle / 'evidence' / name).mkdir(parents=True)
    case = tmp_path / 'Case_ElectroThermal'
    (case / 'Dataset').mkdir(parents=True)
    (case / 'pyproject.toml').touch()
    return bundle, case


def test_public_inputs_do_not_require_private_generation_root(tmp_path):
    bundle, case = _bundle(tmp_path)
    links = link_shared_bundle(bundle, case)
    assert len(links) == 6
    assert link_shared_bundle(bundle, case) == links
    assert (case / 'Dataset/standard.h5').resolve() == bundle / 'standard.h5'


def test_all_public_links_preflight_foreign_and_regular_paths(tmp_path):
    bundle, case = _bundle(tmp_path)
    (case / 'Dataset/pilot').symlink_to(tmp_path / 'foreign')
    with pytest.raises(FileExistsError):
        link_shared_bundle(bundle, case)
    assert not (case / 'Dataset/standard.h5').exists()
    (case / 'Dataset/pilot').unlink()
    (case / 'Dataset/pilot').mkdir()
    with pytest.raises(FileExistsError):
        link_shared_bundle(bundle, case)
    assert not (case / 'Dataset/standard.h5').exists()
