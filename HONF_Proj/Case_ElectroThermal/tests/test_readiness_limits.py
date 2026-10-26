from pathlib import Path

import pytest

from hetmf.readiness import diagnose_releases


@pytest.mark.parametrize("retained,temporary", [(1, 2**30), (2**30, 1)])
def test_readiness_refuses_undersized_budget_before_materialization(tmp_path, monkeypatch, retained, temporary):
    monkeypatch.setattr("hetmf.readiness.require_owned_root", lambda path: Path(path))
    output = tmp_path / "diagnostic"
    with pytest.raises(RuntimeError, match="budget"):
        diagnose_releases(tmp_path / "standard", tmp_path / "expanded", output,
                          retained_budget_bytes=retained, temporary_budget_bytes=temporary)
    assert not output.exists()
