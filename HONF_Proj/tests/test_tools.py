from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_tool(name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"honf_tool_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_inspector_accepts_documented_option(monkeypatch, capsys) -> None:
    module = _load_tool("inspect_dataset")
    monkeypatch.setattr(
        "sys.argv",
        ["inspect_dataset.py", "--dataset-id", "thermal_disk_local_v1"],
    )
    assert module.main() == 0
    assert "thermal_disk_local_v1" in capsys.readouterr().out


def test_dataset_inspector_rejects_conflicting_forms(monkeypatch) -> None:
    module = _load_tool("inspect_dataset")
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect_dataset.py",
            "thermal_disk_local_v1",
            "--dataset-id",
            "thermal_channel_global_v1",
        ],
    )
    with pytest.raises(SystemExit):
        module.main()


def test_evaluate_dispatcher_preserves_repeated_compare_run_ids(monkeypatch) -> None:
    path = Path(__file__).resolve().parents[1] / "evaluate.py"
    spec = importlib.util.spec_from_file_location("honf_evaluate_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--workflow", "compare", "--Run_ID", "0001", "--Run_ID", "0002"],
    )
    args = module.parse_args()
    assert args.run_id is None
    assert args.case_args[-4:] == ["--Run_ID", "0001", "--Run_ID", "0002"]


def test_evaluate_dispatcher_rejects_multiple_single_run_ids(monkeypatch) -> None:
    path = Path(__file__).resolve().parents[1] / "evaluate.py"
    spec = importlib.util.spec_from_file_location("honf_evaluate_entry_single", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--workflow", "forward", "--run-id", "0001", "--run-id", "0002"],
    )
    with pytest.raises(SystemExit):
        module.parse_args()
