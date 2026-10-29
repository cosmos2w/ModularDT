"""The case-plugin registry exposes retained workflows only."""
import pytest

from honf_runtime.registry import MODEL_FAMILIES, require_model_family


@pytest.mark.parametrize("workflow", ["forward", "local_module", "compare"])
def test_legacy_forward_workflows_keep_their_checkpoint_family(workflow):
    assert require_model_family("honf_forward", workflow).available


@pytest.mark.parametrize("workflow", ["forward", "compare"])
def test_content_attention_workflows_keep_their_checkpoint_family(workflow):
    assert require_model_family("honf_content_hypergraph", workflow).available


@pytest.mark.parametrize("family", ["joint_native_hypergraph_v1", "inherited_attention_v1",
    "moment_hyperedges_round5", "hyperedge_readout_round6", "native_retrieval_v1",
    "native_budgeted_dispatch_v2"])
def test_retired_workflows_are_not_selectable(family):
    with pytest.raises(ValueError, match="Unknown model_family"):
        require_model_family(family, "forward")


def test_inverse_namespace_remains_reserved():
    assert set(MODEL_FAMILIES) == {"honf_forward", "honf_content_hypergraph", "honf_inverse"}
    with pytest.raises(NotImplementedError):
        require_model_family("honf_inverse", "inverse")
