"""The cache verification command cannot silently build or repair a miss."""
import numpy as np
import pytest

import wind_native_catalogue_cache as cli


def test_verify_passes_strict_read_only_policy_and_propagates_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "WindFarmNativeView", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_load_original_split", lambda *args: object())
    monkeypatch.setattr(cli, "select_rows", lambda *args: (np.array([1]), {}))
    actual = {}

    class Cache:
        def __init__(self, **kwargs):
            actual.update(kwargs)

    def check(view, rows, cache, **kwargs):
        assert actual["read_only_persistent"] is True
        assert actual["require_prebuilt"] is True
        raise FileNotFoundError("missing prebuilt geometry")

    monkeypatch.setattr(cli, "NativeRoleCatalogueCache", Cache)
    monkeypatch.setattr(cli, "prepare_rows", check)
    with pytest.raises(FileNotFoundError, match="missing prebuilt"):
        cli.main(["--scope", "full", "--verify", "--cache-dir", str(tmp_path)])


def test_prepare_and_verify_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.main(["--scope", "full", "--verify", "--prepare"])
