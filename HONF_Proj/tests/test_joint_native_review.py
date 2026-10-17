from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from honf_forward_core.joint_native.model import JointNativeCore
from honf_runtime.joint_native_review import JointNativeTrainReview
from tests.test_content_hypergraph_v2_core import _core
from tests.test_native_budgeted_protocol import _Provider,_engine


@pytest.mark.parametrize("mode",["joint","direct"])
def test_fixed_train_review_is_inert_checkpointed_and_replays_retention_cliff(tmp_path,mode):
    provider=_Provider();provider.training_samples=7
    engine=_engine();engine.epoch_case_order=lambda provider,epoch:(["fixture-case"],"fixture-native")
    engine.prepare_update_microbatches=lambda *args,**kwargs:[provider.batch]
    model=JointNativeCore.from_a2d(_core("A2-D"),routing_mode=mode).train()
    model.joint_begin_update(3025,0)
    manager=JointNativeTrainReview()
    layout=SimpleNamespace(write_path=lambda name,category:tmp_path/name)
    before={name:value.detach().clone() for name,value in model.state_dict().items() if torch.is_tensor(value)}
    rng=torch.random.get_rng_state().clone()
    result=manager.review_epoch(model,provider,engine,3025,layout,training_row={"train_losses":{"native_field":3.}})
    assert result["epoch"]==3025 and provider.training_samples==7 and model.training
    assert torch.equal(torch.random.get_rng_state(),rng)
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())
    restored=JointNativeTrainReview();restored.load_state_dict(manager.state_dict())
    with torch.no_grad():
        if mode=="joint":
            for route in model.router.graph_parameters.routes.values():route.bias.fill_(100)
        else:
            model.router.direct_parameters.module.bias.fill_(100)
            model.router.direct_parameters.environment.bias.fill_(100)
    model.joint_begin_update(3050,0)
    result=restored.review_epoch(model,provider,engine,3050,layout,training_row={"train_losses":{"native_field":2.}})
    payload=json.loads(Path(result["path"]).read_text())
    assert payload["retention_cliff_types"]
    assert payload["fixed_batch_replay"]["support_and_native_terms_repeat_exactly"]
    assert payload["native_epoch_loss_terms"]["native_field"]["delta"]==-1
    assert payload["duals"]["M"]["phase"]=="stochastic_dense_router"
    assert payload["changed_original_ID_support"]["0:0"]["E_support"]["changed"]>0
    assert payload["changed_original_ID_support"]["0:0"]["E_support"]["changed_ID_examples"]
    with pytest.raises(ValueError,match="advance"):
        restored.review_epoch(model,provider,engine,3050,layout)


def test_review_phase_and_cap_occupancy_are_serialized():
    from honf_forward_core.joint_native.budget import JointBudgetState
    state=JointBudgetState(.1,.2);state.set_clock(3371,4)
    assert state.duals["M"].phase=="balanced_stochastic_router"
    for step in range(1,31):state.observe_residuals(torch.tensor(100.),torch.tensor(-1.),step)
    restored=JointBudgetState();restored.load_state_dict(state.state_dict())
    assert restored.duals["M"].phase=="balanced_stochastic_router"
    assert int(restored.duals["M"].observed_macro_count)==30
    assert int(restored.duals["M"].at_cap_macro_count)==21
    assert int(restored.duals["E"].at_cap_macro_count)==0
