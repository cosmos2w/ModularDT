"""Public research CLI retains anchored aliases and rejects retired arms."""
import importlib.util
from pathlib import Path

import pytest

from honf_runtime.contextual_frontier_evaluation import _contextual_checkpoint_parts


PROJECT = Path(__file__).resolve().parents[1]


def _cli():
    spec = importlib.util.spec_from_file_location("anchored_research_cli", PROJECT / "tools/honf_research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["evaluate-r13-8", "evaluate-member-routing"])
def test_evaluation_aliases_select_same_anchored_evaluator(name, monkeypatch):
    from honf_runtime import r13_8_anchored_organization_evaluation as evaluation
    calls = []
    monkeypatch.setattr(evaluation, "evaluate", lambda *args, **kw: calls.append((args, kw)))
    assert _cli().main([name, "--checkpoint", "selected.pt", "--output", "result", "--representative-case", "case1"]) == 0
    assert calls == [(("selected.pt", "cpu", "result"), {"representative_case_ids": ["case1"]})]


@pytest.mark.parametrize("name", ["train-contextual", "continue-contextual", "resume-contextual", "evaluate-joint-member-routing", "evaluate-r13-6-member-routing"])
def test_retired_commands_are_not_public(name):
    with pytest.raises(SystemExit):
        _cli()._parser().parse_args([name])


def test_evaluator_rejects_unretained_workflow():
    with pytest.raises(ValueError, match="anchored"):
        _contextual_checkpoint_parts({"experiment_identity": {"workflow": "contextual_frontier_joint_training_v1"}})
