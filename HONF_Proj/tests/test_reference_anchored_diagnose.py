import importlib.util
import copy
from pathlib import Path

import pytest
import torch

from honf_runtime.unified_training import LossTerm


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/diagnostics/reference_anchored_diagnose.py"
_SPEC = importlib.util.spec_from_file_location("reference_anchored_diagnose_objective", _MODULE_PATH)
_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_cli)


class _ValidationSubsetProvider:
    def loss_metadata(self):
        return {name: {"weight": "source"} for name in ("temperature/fluid", "response", "operator_residual")}

    def validation_loss_terms(self, predictions, targets, auxiliary_state, *, batch, arm):
        # Response and stored operator supervision are training-only in this provider.
        return {"temperature/fluid": LossTerm(torch.tensor(2.0), 1.0, 0.5)}

    def loss_terms(self, predictions, targets, phase, auxiliary_state):
        return {"temperature/fluid": LossTerm(torch.tensor(2.0), 1.0, 0.5)}


def test_sparse_conditional_overrides_apply_when_present_and_allow_absent_validation_terms():
    overrides = {"temperature/fluid": 0.0, "response": 0.0, "operator_residual": 0.0}
    provider = _cli._DiagnosticObjectiveProvider(_ValidationSubsetProvider(), ["0002"], overrides)

    validation = provider.validation_loss_terms(None, None, None, batch=None, arm="P")
    assert set(validation) == {"temperature/fluid"}
    assert validation["temperature/fluid"].weight == 0.0

    # The same provider can emit response/operator terms only on eligible TRAIN
    # microbatches; an absent conditional term contributes zero for that call.
    training = provider.loss_terms(None, None, "joint", None)
    assert set(training) == {"temperature/fluid"}
    assert training["temperature/fluid"].weight == 0.0

    with pytest.raises(ValueError, match="names no declared provider loss term"):
        _cli._DiagnosticObjectiveProvider(_ValidationSubsetProvider(), ["0002"], {"typo": 0.0})


def test_fit_receipt_resolves_the_exact_epoch_checkpoint_from_run_layout(tmp_path):
    output = tmp_path / "diagnostic-child"
    checkpoint = output / "checkpoints" / "epoch_1025_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"review endpoint")

    resolved, digest = _cli._diagnostic_result_checkpoint(output, 1025)

    assert resolved == checkpoint.resolve()
    assert digest == _cli._sha256(checkpoint)


def test_diagnostic_resume_requires_the_run_directory_latest_checkpoint(tmp_path):
    output = tmp_path / "diagnostic-child"
    latest = output / "checkpoints" / "latest_model.pt"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"exact latest")

    assert _cli._validate_diagnostic_fit_output(output, latest) == output.resolve()
    with pytest.raises(ValueError, match="exact latest checkpoint"):
        _cli._validate_diagnostic_fit_output(output, tmp_path / "other.pt")


def test_diagnostic_wrapper_cloned_probe_uses_native_provider_model_and_restores_source():
    """A native identity check and backward probe; no optimizer update."""
    class NativeProvider(_ValidationSubsetProvider):
        def predict(self, model):
            if model is not self.model:
                raise ValueError("Native provider model identity changed.")
            return model(torch.ones(1, 1)).sum()

    source = torch.nn.Linear(1, 1)
    native = NativeProvider()
    native.model = source
    wrapped = _cli._DiagnosticObjectiveProvider(native, ["0002"], {})
    clone = copy.deepcopy(source)
    original = wrapped.model
    try:
        wrapped.model = clone
        wrapped.predict(clone).backward()
        assert clone.weight.grad is not None
        assert torch.equal(clone.weight.grad, torch.ones_like(clone.weight))
        assert source.weight.grad is None
    finally:
        wrapped.model = original
    assert wrapped.model is source
    assert native.model is source
