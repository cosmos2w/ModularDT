"""Joint lifecycle preserves ordinary updates, validation, and exact resume."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_unified_training import _config, _ToyProvider

from honf_runtime.compat import load_trusted_checkpoint
from honf_runtime.run_layout import resolve_checkpoint
from honf_runtime.unified_training import TrainingEngine, _engine_config_payload


def _joint_config():
    return replace(_config(), training_mode="joint", warmup_epochs=0,
                   open_through_epoch=0, soft_through_epoch=0,
                   sampling_version="case_epoch_v1")


def test_joint_exact_resume_and_validation_keep_all_heads_active(tmp_path, monkeypatch):
    import honf_runtime.unified_training as runtime
    monkeypatch.setattr(runtime, "_render_loss_curves", lambda *args, **kwargs: None)
    torch.manual_seed(71)
    initial = torch.nn.Linear(1, 1).state_dict()

    class JointProvider(_ToyProvider):
        def __init__(self):
            super().__init__()
            self.dev_grad_states = []

        def predict_native(self, model, *args, **kwargs):
            if not model.training:
                self.dev_grad_states.append(torch.is_grad_enabled())
            return super().predict_native(model, *args, **kwargs)

    def run(directory, stop, resume=False):
        model = torch.nn.Linear(1, 1)
        model.load_state_dict(initial)
        provider = JointProvider()
        engine = TrainingEngine(_joint_config(), device="cpu")
        engine.fit(model, provider, directory, identity={"workflow": "joint_test"},
                   arm="J-H", stop_after=stop,
                   resume_checkpoint=resolve_checkpoint(directory, "latest") if resume else None)
        assert all(phase[0] == "joint" for phase in provider.validation_phases)
        assert provider.dev_grad_states and not any(provider.dev_grad_states)
        assert torch.is_grad_enabled()
        return load_trusted_checkpoint(resolve_checkpoint(directory, "latest"), map_location="cpu")

    complete = run(tmp_path / "complete", 10)
    run(tmp_path / "resumed", 4)
    resumed = run(tmp_path / "resumed", 10, True)
    assert complete["workflow"] == "joint_test"
    for name, value in complete["model_state_dict"].items():
        torch.testing.assert_close(value, resumed["model_state_dict"][name], rtol=0, atol=0)
    for name, state in complete["optimizer_state_by_name"]["state_by_name"].items():
        for key, value in state.items():
            other = resumed["optimizer_state_by_name"]["state_by_name"][name][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
    assert sum(row["optimizer_updates"] for row in resumed["history"]) == 20
    assert all(row["phase"] == "joint" for row in resumed["history"])


def test_joint_mode_is_explicit_and_legacy_resume_identity_unchanged():
    legacy = _engine_config_payload(_config())
    assert "training_mode" not in legacy
    assert _engine_config_payload(_joint_config())["training_mode"] == "joint"
    with pytest.raises(ValueError, match="Training mode"):
        replace(_config(), training_mode="unknown")


def _cli():
    path = Path(__file__).parents[1] / "tools/joint_regional_train.py"
    spec = importlib.util.spec_from_file_location("joint_regional_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", ["thermal", "wind"])
def test_formal_recipe_rejects_external_models_and_implicit_launch(task):
    module = _cli()
    path = Path(__file__).parents[1] / f"src/config_core/forward/joint_regional/{task}_full5000_v1.json"
    recipe = module.read_recipe(path)
    assert recipe["total_epochs"] == 5000 and recipe["initialization"] == "fresh_all_trainable"
    with pytest.raises(ValueError, match="unsupported"):
        module.validate_recipe({**recipe, "flow_checkpoint": "old_D_sep.pt"})
    with pytest.raises(SystemExit):
        module.main(["start", "--recipe-json", str(path), "--output-dir", "/unused", "--stop-after", "5000"])
