"""Published source paths, curated figures, and artifact exclusions."""
from pathlib import Path
import runpy
import re

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_project_paths_are_root_relative_and_independent_of_cwd(tmp_path, monkeypatch):
    from honf_runtime.paths import PROJECT_ROOT, resolve_path
    from honf_runtime.config_loader import load_config_bundle
    monkeypatch.chdir(tmp_path)
    assert PROJECT_ROOT == ROOT
    path = resolve_path('project://src/config_core/forward/stage7_structured_context.json')
    assert path == ROOT/'src/config_core/forward/stage7_structured_context.json'
    bundle = load_config_bundle(path)
    assert bundle.effective['case']['id'] == 'ThermalChannel'


def test_homepage_images_are_portable_decodable_and_not_upscaled():
    text = (ROOT/'README.md').read_text()
    images = re.findall(r'<img src="([^"]+)" alt="([^"]+)" width="(\d+)"', text)
    assert len(images) == 8
    for rel, alt, width in images:
        assert rel.startswith('docs/assets/readme/')
        assert alt.strip()
        path = ROOT/rel
        with Image.open(path) as image:
            assert int(width) <= image.width
            image.verify()
        assert path.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    assert 'docs/reports/' not in text
    assert 'diagnostics/' not in text


@pytest.mark.parametrize('path,blocked', [
    ('docs/assets/readme/model_fields.png',False),
    ('docs/assets/readme/model_fields.pdf',False),
    ('docs/assets/readme/numerical_source.npz',True),
    ('docs/assets/readme_backup/model_fields.png',True),
    ('docs/reports',True),
    ('docs/reports/README.md',True),
    ('UpgradePlan',True),
    ('UpgradePlan/plan.md',True),
    ('HONF_Proj/docs/reports/report.md',True),
    ('diagnostics/receipt.json',True),
    ('tools/diagnostics/run_specific.py',True),
    ('docs/assets/readme/numerical_source.json',True),
    ('src/honf_forward_core/models.py',False),
])
def test_artifact_hook_has_only_the_approved_figure_exception(path,blocked):
    policy = runpy.run_path(str(ROOT/'.githooks/pre-push'))
    assert policy['forbidden'](path) is blocked
