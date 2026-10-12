"""Architecture lineage guards; optimizer fixtures are never fitted parents."""
import copy
import hashlib
from dataclasses import asdict

import pytest
import torch
from test_unified_training import _ToyProvider
from torch import nn

import honf_runtime.unified_training as runtime
from honf_runtime.architecture_child import (
    SCHEMA,
    architecture_child_provider_state,
    prepare_native_routing_child,
)
from honf_runtime.run_layout import RunLayout


class _Model(nn.Module):
    def __init__(self, mode="P"):
        super().__init__()
        self.core = nn.Module()
        self.core.field = nn.Linear(1, 1)
        self.mode = mode
        if mode != "P":
            self.core.native_routing = nn.Module()
            output = 1
            modules = nn.ModuleList([
                nn.Sequential(nn.Linear(1, 32), nn.SiLU(),
                              nn.Linear(32, output, bias=mode == "P-C")) for _ in range(3)])
            if mode == "P-R":
                self.core.native_routing.score_mlps = modules
            else:
                self.core.native_routing.aggregate_modulators = modules
            for module in modules:
                nn.init.zeros_(module[2].weight)
                if module[2].bias is not None:
                    nn.init.zeros_(module[2].bias)

    def forward(self, value):
        result = self.core.field(value)
        if self.mode != "P":
            modules = next(iter(self.core.native_routing.children()))
            result = result + sum(module(value) for module in modules)
        return result


class _Provider(_ToyProvider):
    def __init__(self, mode="P"):
        super().__init__()
        self.mode = mode

    def identity_payload(self):
        family = ("thermal_interaction_preserving_native_curl_v1" if self.mode == "P"
                  else "thermal_native_curl_interaction_routing_v1")
        config = {"family": family, "mode": self.mode, "regional_anchors": 0, "seed": 0}
        if self.mode != "P":
            config["native_routing"] = {
                "schema": "native_interaction_routing_v1", "mode": self.mode,
                "initialization_seed": 0,
                "policy": "learned_density" if self.mode == "P-R" else "destination_gain",
                **({"score_hidden": 32, "score_bound": 2.0} if self.mode == "P-R"
                   else {"control_hidden": 32})}
        return {"task": "ThermalChannel", "model_family": family, "arm": self.mode,
                "model_config": config, "training_case_ids": list(range(150)),
                "manifest_fingerprint": "fixed", "loss_weights": {"native": 1.0}}

    def epoch_cases(self, epoch, seed):
        return tuple(range(150))

    def optimizer_groups(self, model, arm, stage):
        return (runtime.OptimizerGroupSpec(
            name="joint", parameter_names=tuple(name for name, _ in model.named_parameters()),
            schedule=runtime.ScheduleSpec(peak_lr=.0003, warmup_epochs=20,
                hold_through_epoch=1000, total_epochs=2500, final_lr=.000003), weight_decay=.0001),)


@pytest.mark.parametrize("mode", ["P-R", "P-C"])
def test_child_provider_state_maps_only_arm_without_mutating_parent(mode):
    source = {"arm": "P", "training_mode": "joint", "response_coefficient": .01,
              "operator_coefficient": .02, "auxiliary_calibration": {"sha256": "fixed"}}
    class Provider:
        def training_state_dict(self):
            return {**copy.deepcopy(source), "arm": mode}
    mapped = architecture_child_provider_state(Provider(), source, mode)
    assert mapped == {**source, "arm": mode}
    assert source["arm"] == "P"
    mapped["auxiliary_calibration"]["sha256"] = "changed"
    assert source["auxiliary_calibration"]["sha256"] == "fixed"


@pytest.mark.parametrize("change", ["arm", "calibration", "coefficient", "extra"])
def test_child_provider_state_rejects_non_arm_changes_without_steps(change):
    source = {"arm": "P", "coefficient": 1., "calibration": {"sha256": "fixed"}}
    current = {**copy.deepcopy(source), "arm": "P-R"}
    if change == "arm": source["arm"] = "P-H"
    elif change == "calibration": current["calibration"]["sha256"] = "changed"
    elif change == "coefficient": current["coefficient"] = .5
    else: current["new_recipe"] = True
    class Provider:
        def training_state_dict(self):
            return current
    with pytest.raises(ValueError):
        architecture_child_provider_state(Provider(), source, "P-R")


def test_child_wind_empty_provider_state_remains_empty():
    assert architecture_child_provider_state(object(), {}, "P-C") == {}


def _source(tmp_path, mode):
    config = runtime.EngineConfig(seed=0, microbatch_cases=8, effective_cases=150,
        total_epochs=2500, warmup_epochs=20, open_through_epoch=20, soft_through_epoch=20,
        monitor_every=100, monitor_epochs=(1001, 1002), checkpoint_epochs=(1001, 1002),
        latest_every=100, curve_every=100, sampling_version="case_epoch_v1", training_mode="joint")
    engine = runtime.TrainingEngine(config, device="cpu")
    model, provider = _Model(), _Provider()
    optimizer, specs = engine._make_optimizer(model, provider, "P", 1000, None)
    for p in model.parameters():
        optimizer.state[p] = {"step": torch.tensor(4000.), "exp_avg": torch.full_like(p,.01),
                              "exp_avg_sq": torch.full_like(p,.1)}
    identity = {"workflow": "unit_lineage", "provider_identity": provider.identity_payload(),
                "engine_config": runtime._engine_config_payload(config),
                "selection_policy": asdict(engine.selection),
                "optimizer_schedule_contract": [runtime._optimizer_spec_payload(s) for s in specs],
                "sampling_dataset_id": runtime._sampling_dataset_id(provider.identity_payload())}
    payload = engine._checkpoint_payload(model,optimizer,identity=identity,arm="P",epoch=1000,
        history=[{"epoch":1000,"case_visits":150000,"optimizer_updates":4000}],
        best_field=1.,best_guarded=1.,sampler_state={},group_specs=specs)
    path=tmp_path/"source.pt"; torch.save(payload,path)
    child=_Model(mode); child_provider=_Provider(mode)
    child_identity={**copy.deepcopy(identity),"provider_identity":child_provider.identity_payload(),
        "optimizer_schedule_contract":[runtime._optimizer_spec_payload(s) for s in child_provider.optimizer_groups(child,mode,"joint")]}
    declaration={"schema":SCHEMA,"source_checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_epoch":1000,"source_arm":"P","target_mode":mode,
        "added_parameter_names":sorted(set(child.state_dict())-set(model.state_dict())),
        "review_epochs":[1001,1002],"final_epoch":1002,"scientific_update_limit":2,
        "reason":"Unit test of exact source/moment mapping and child-only work accounting."}
    return engine,payload,path,child,child_provider,child_identity,declaration


@pytest.mark.parametrize("change",["cohort","loss","seed","schedule","selector","extra_parameter","nonzero_final","mode","budget","source"])
def test_architecture_child_rejects_contract_changes_before_stepping(tmp_path,change):
    _engine,payload,path,child,_provider,identity,declaration=_source(tmp_path,"P-R")
    if change=="cohort": identity["provider_identity"]["training_case_ids"][0]=999
    elif change=="loss": identity["provider_identity"]["loss_weights"]["native"]=.5
    elif change=="seed": identity["engine_config"]["seed"]=3
    elif change=="schedule": identity["optimizer_schedule_contract"][0]["weight_decay"]=.2
    elif change=="selector": identity["selection_policy"]["field_metric"]="changed"
    elif change=="extra_parameter": child.extra=nn.Parameter(torch.tensor(1.))
    elif change=="nonzero_final": child.core.native_routing.score_mlps[0][2].weight.data.fill_(1.)
    elif change=="mode": declaration["target_mode"]="P-H"
    elif change=="budget": declaration["scientific_update_limit"]=1
    elif change=="source": declaration["source_checkpoint_sha256"]="bad"
    with pytest.raises((ValueError,TypeError)):
        prepare_native_routing_child(child,payload,identity,declaration,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),case_keys=tuple(range(150)),stop_after=1001)


@pytest.mark.parametrize("mode",["P-R","P-C"])
def test_architecture_child_and_exact_resume_keep_old_moments_and_new_age(tmp_path,monkeypatch,mode):
    """Exactly two disposable optimizer updates per mode; four in total."""
    monkeypatch.setattr(runtime,"_render_loss_curves",lambda *a,**k:None)
    engine,payload,path,child,provider,identity,declaration=_source(tmp_path,mode)
    output=tmp_path/"child"
    engine.fit(child,provider,output,identity=identity,arm=mode,stop_after=1001,
        branch_from_checkpoint=path,architecture_child_initialization=declaration)
    latest=RunLayout(output).read_path("latest_model.pt")
    checkpoint=torch.load(latest,map_location="cpu",weights_only=False)
    receipt=checkpoint["branch_parent"]["architecture_child_initialization"]
    assert receipt["exact_inherited_moments_verified"] is True
    assert receipt["new_optimizer_states_at_attachment"]==0
    assert sum(row["optimizer_updates"] for row in checkpoint["history"])==1
    assert checkpoint["branch_parent"]["inherited_optimizer_updates"]==4000
    states=checkpoint["optimizer_state_by_name"]["state_by_name"]
    assert all(int(states[n]["step"])==1 for n in declaration["added_parameter_names"])
    assert all(int(states[n]["step"])==4001 for n in payload["model_state_dict"])
    engine.fit(child,provider,output,identity=identity,arm=mode,stop_after=1002,resume_checkpoint=latest)
    resumed=torch.load(latest,map_location="cpu",weights_only=False)
    assert sum(r["optimizer_updates"] for r in resumed["history"])==2
    assert all(int(resumed["optimizer_state_by_name"]["state_by_name"][n]["step"])==2
               for n in declaration["added_parameter_names"])
    with pytest.raises(ValueError,match="sealed reviews"):
        engine.fit(child,provider,output,identity=identity,arm=mode,stop_after=1003,resume_checkpoint=latest)
