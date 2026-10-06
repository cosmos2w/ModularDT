"""Refinement scope/lineage cannot silently broaden historical interface fitting."""

import copy
import hashlib
from pathlib import Path

import pytest
import torch
from channelthermal.training.interface_fit import (
    FIT_KIND,
    FIT_PARAMETER_PREFIX,
    freeze_interface_fit_model,
    validate_interface_fit_campaign,
)
from channelthermal.training.optimizer import build_forward_optimizer
from channelthermal.training.refinement_policy import (
    PHYSICAL_PREFIXES,
    POLICY_DECLARATION,
    REFINEMENT_KIND,
    apply_refinement_policy,
    assert_refinement_frozen_state,
    attach_refinement_checkpoint,
    capture_refinement_frozen_state,
    refinement_parameter_inventory,
    validate_refinement_parent,
    validate_refinement_resume,
    validate_refinement_trainable_inventory,
)
from torch import nn

from tests.test_interface_fit import Config
from tests.test_interface_fit import fixture as fit_fixture
from tests.test_tensor_query_interaction import field, nonzero
from tests.test_tensor_source_group_residual import read, record

LABEL_DIRECTORY = Path("/response")


@pytest.fixture(autouse=True)
def policy_unit_label_identity(tmp_path, monkeypatch):
    """Data-loader provenance is tested separately; use real disposable bytes here."""
    import channelthermal.training.response_refinement as response

    monkeypatch.setitem(globals(), "LABEL_DIRECTORY", tmp_path)
    for anchor in ("0001", "0318", "0333", "0348"):
        (tmp_path / f"{anchor}.npz").write_bytes(b"unit response numerical bytes")
        (tmp_path / f"{anchor}.json").write_bytes(b"unit response provenance bytes")
    def identity(declaration):
        return [{"atlas_npz_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                 "atlas_json_sha256": hashlib.sha256(Path(path).with_suffix(".json").read_bytes()).hexdigest()}
                for path in declaration["response_addendum"]["response_stencils"]]
    monkeypatch.setattr(response, "response_addendum_identity", identity)


class Host(nn.Module):
    """Small materialized wrapper with every actual semantic parameter path."""

    def __init__(self):
        super().__init__()
        for prefix in (FIT_PARAMETER_PREFIX,) + PHYSICAL_PREFIXES + (
                "core.common.coarse_builder.", "core.common.context_norm.", "core.module_feature_encoder.",
                "core.backend.organizer.", "core.backend.control_gain.", "local_coupling.local_surrogate."):
            pieces = prefix.rstrip(".").split(".")
            parent = self
            for part in pieces[:-1]:
                if not hasattr(parent, part): setattr(parent, part, nn.Module())
                parent = getattr(parent, part)
            setattr(parent, pieces[-1], nn.Linear(2, 2))
        self.register_buffer("normalizer", torch.tensor([3.]))
        self.register_buffer("nonpersistent_scale", torch.tensor(2.), persistent=False)
        self.register_buffer("_selection_epoch_state", torch.tensor(100))
        self.core.backend.tensor_query_interaction.admission_mode = "soft"

    def set_training_progress(self, **kwargs):
        self.progress = kwargs
        self._selection_epoch_state.fill_(kwargs["epoch"])

    def forward(self, x):
        x = self.core.module_feature_encoder(x)
        for prefix in (FIT_PARAMETER_PREFIX,) + PHYSICAL_PREFIXES:
            component = self
            for part in prefix.rstrip(".").split("."): component = getattr(component, part)
            x = x + .1 * torch.tanh(component(x))
        return x


def fixture():
    _, _, fit = fit_fixture()
    source = Host()
    frozen_names = freeze_interface_fit_model(source)
    parent = {"checkpoint_schema_version": 1, "case_id": "ThermalChannel", "model_family": "honf_forward",
              "workflow": "forward", "epoch": 100, "selection_state": {"epoch": 100, "total_epochs": 1000},
              "model_config": copy.deepcopy(fit["model"]), "train_config": copy.deepcopy(fit),
              "model_state_dict": copy.deepcopy(source.state_dict()), "optimizer_state_dict": {"state": {0: "old moment"}},
              "rng_state": {"torch": torch.get_rng_state()},
              "global_normalization_stats": {"mean": torch.tensor([2.])},
              "local_normalization_stats": {"mean": torch.tensor([4.])},
              "campaign_training_state": {"interface_fit_attachment": {"kind": FIT_KIND,
                  "inherited_backbone_epoch": 1000, "new_parameter_names": frozen_names}, "response_scale": .1}}
    config = copy.deepcopy(fit)
    config["model"]["core_honf"]["interface_model"]["hypergraph_options"]["query_admission_mode"] = "soft"
    config["training"]["learning_rate"] = 1e-5
    campaign = config["training"]["campaign"]
    campaign.pop("interface_fit")
    campaign["forward_refinement"] = copy.deepcopy(POLICY_DECLARATION)
    campaign["parent"] = {"kind": REFINEMENT_KIND, "epoch": 100, "checkpoint": "/H-add/epoch_100_model.pt"}
    campaign["response_addendum"] = {"policy_version": "train_heat_families_v1",
        "fit_anchor_ids": ["0001", "0318", "0333", "0348"],
        "response_development_anchor_ids": ["0304", "0320", "0335", "0350"],
        "fixed_audit_anchor_ids": ["0277", "0291", "0294", "0687"],
        "variant_labels": ["heat_transfer_minus", "heat_transfer_plus"],
        "response_stencils": [str(LABEL_DIRECTORY / f"{anchor}.npz") for anchor in ["0001", "0318", "0333", "0348"]]}
    campaign["response_refinement"] = {"auxiliary_absolute_weight": .25, "q_proxy_weight": .25,
        "response_floor_fraction": .001, "heat_null_capability": "thermalchannel_analytic_heat_independent_flow_v1",
        "null_fractions": [.05, .10], "null_fluid_queries": 256, "sampling_seed": 0,
        "calibration": {"response_coefficient": .1, "null_coefficient": .1}}
    target = Host(); target.config = Config(config["model"])
    return parent, target, config


def attach(parent, target, config):
    return attach_refinement_checkpoint(parent, target_model=target, target_config=target.config,
        refinement_config=config, source_path=config["training"]["campaign"]["parent"]["checkpoint"], source_sha256="digest")


def test_exact_policy_separates_historical_frozen_host_and_physical_groups():
    _, model, config = fixture()
    freeze_interface_fit_model(model)
    assert all(name.startswith(FIT_PARAMETER_PREFIX) for name, parameter in model.named_parameters() if parameter.requires_grad)
    with pytest.raises(ValueError, match="inventory"): build_forward_optimizer(model, config["training"])
    inventory = apply_refinement_policy(model)
    optimizer, groups = build_forward_optimizer(model, config["training"])
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-5]
    assert all(group["weight_decay"] == 1e-5 for group in optimizer.param_groups)
    assert groups["groups"][0]["parameter_names"] == inventory["parameter_names"]["organizer"]
    assert groups["groups"][1]["parameter_names"] == inventory["parameter_names"]["physical"]
    assert inventory["materialized"] and inventory["scalar_counts"]["physical"] > 0
    model.core.common.context_norm.weight.requires_grad = True
    with pytest.raises(ValueError, match="inventory"): validate_refinement_trainable_inventory(model)


def test_attachment_replays_own_parent_preserves_normalizers_and_discards_old_optimizer():
    parent, model, config = fixture()
    child, receipt = attach(parent, model, config)
    source = Host(); source.load_state_dict(parent["model_state_dict"])
    x = torch.tensor([[.2, .4], [.1, .3]], requires_grad=True)
    torch.testing.assert_close(model(x), source(x), rtol=0, atol=0)
    assert child["epoch"] == 0 and model.progress == {"epoch": 0, "total_epochs": 1000}
    assert receipt["inherited_backbone_epoch"] == 1000 and receipt["interface_prefit_epoch"] == 100
    assert child["optimizer_state_dict"]["state"] == {}
    assert "response_scale" not in child["campaign_training_state"]
    assert child["campaign_training_state"]["previous_interface_fit_training_state"]["response_scale"] == .1
    for key in ("global_normalization_stats", "local_normalization_stats"):
        torch.testing.assert_close(child[key]["mean"], parent[key]["mean"], rtol=0, atol=0)
    validate_refinement_resume(child, model, config["training"]["campaign"], config)
    frozen = capture_refinement_frozen_state(model)
    optimizer, _ = build_forward_optimizer(model, config["training"])
    before = copy.deepcopy(model.state_dict())
    model(x).square().sum().backward()
    assert x.grad is not None and torch.count_nonzero(x.grad)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for _, parameter in model.named_parameters() if parameter.requires_grad)
    optimizer.step()
    model.set_training_progress(epoch=100, total_epochs=1000)
    assert_refinement_frozen_state(model, frozen)
    inventory = refinement_parameter_inventory(model)
    for group in ("organizer", "physical"):
        assert any(not torch.equal(before[name], model.state_dict()[name]) for name in inventory["parameter_names"][group])


@pytest.mark.parametrize("change", ["fit300", "other_arm", "missing_rng", "refinement_parent", "formal", "identity"])
def test_parent_rejects_wrong_lineage(change):
    parent, _, config = fixture()
    if change == "fit300": parent["epoch"] = 300
    elif change == "other_arm": parent["model_config"]["core_honf"]["interface_model"]["hypergraph_options"]["query_interaction_mode"] = "joint"
    elif change == "missing_rng": parent["rng_state"] = None
    elif change == "refinement_parent": parent["campaign_training_state"]["forward_refinement_attachment"] = {"kind": REFINEMENT_KIND}
    elif change == "formal": parent["train_config"]["dataset"]["development_subset"] = None
    else: parent["case_id"] = "WindFarm"
    with pytest.raises(ValueError): validate_refinement_parent(parent, config)


@pytest.mark.parametrize("change", ["policy", "addendum", "data_bytes", "age", "soft", "runtime_soft", "physical_scope", "dataset", "normalizer", "frozen", "loss"])
def test_resume_rejects_silent_scientific_changes(change):
    parent, model, config = fixture(); child, _ = attach(parent, model, config)
    campaign = config["training"]["campaign"]
    if change == "policy": campaign["forward_refinement"]["physical_learning_rate"] *= 2
    elif change == "addendum": campaign["response_addendum"]["fit_anchor_ids"][0] = "0291"
    elif change == "data_bytes": Path(campaign["response_addendum"]["response_stencils"][0]).write_bytes(b"silently substituted response labels")
    elif change == "age": child["campaign_training_state"]["forward_refinement_attachment"]["refinement_epoch"] = 100
    elif change == "soft": model.config.value["core_honf"]["interface_model"]["hypergraph_options"]["query_admission_mode"] = "curriculum"
    elif change == "runtime_soft": model.core.backend.tensor_query_interaction.admission_mode = "curriculum"
    elif change == "physical_scope": model.core.backend.organizer.weight.requires_grad = True
    elif change == "dataset": config["dataset"]["extra_membership"] = ["0291"]
    elif change == "normalizer": model.nonpersistent_scale.add_(1)
    elif change == "frozen":
        with torch.no_grad(): model.core.module_feature_encoder.weight.add_(1)
    else: config["loss"]["field_weight"] = 2.
    with pytest.raises(ValueError): validate_refinement_resume(child, model, campaign, config)


def test_resumed_optimizer_step_matches_uninterrupted_step_and_run_length_extension_is_allowed():
    parent, model, config = fixture(); child, _ = attach(parent, model, config)
    optimizer, _ = build_forward_optimizer(model, config["training"])
    x = torch.tensor([[.2, .4]])
    def step(target, opt):
        opt.zero_grad(); target(x).square().sum().backward(); opt.step()
    step(model, optimizer)
    child["model_state_dict"] = copy.deepcopy(model.state_dict())
    child["optimizer_state_dict"] = copy.deepcopy(optimizer.state_dict())
    resumed = Host(); resumed.config = Config(config["model"])
    resumed.load_state_dict(child["model_state_dict"]); apply_refinement_policy(resumed)
    config["training"]["epochs"] = 500
    validate_refinement_resume(child, resumed, config["training"]["campaign"], config)
    resumed_optimizer, _ = build_forward_optimizer(resumed, config["training"])
    resumed_optimizer.load_state_dict(child["optimizer_state_dict"])
    step(model, optimizer); step(resumed, resumed_optimizer)
    for name, value in model.state_dict().items(): torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


def test_missing_semantic_component_is_rejected_before_launch():
    _, model, _ = fixture(); del model.local_coupling.port_refinement_head
    with pytest.raises(ValueError, match="components missing"): apply_refinement_policy(model)


def test_historical_fit_cannot_silently_opt_into_fixed_soft_refinement():
    _, _, config = fit_fixture()
    config["model"]["core_honf"]["interface_model"]["hypergraph_options"]["query_admission_mode"] = "soft"
    with pytest.raises(ValueError, match="frozen development"):
        validate_interface_fit_campaign(config["training"]["campaign"], config)


def test_soft_mode_donor_and_admission_are_age_independent_with_real_forward_backward():
    model = field(); encoded, query = record(), torch.rand(2, 5, 2, dtype=torch.float64)
    nonzero(model)
    model.tensor_query_interaction.admission_mode = "soft"
    reference = None
    for age in (0, 100, 150, 500, 1000):
        model.set_training_progress(epoch=age)
        plan = model.tensor_query_interaction.build_plan(encoded)
        assert plan.additional_age == age and plan.sparse_fraction == 0
        assert torch.all(plan.admission[plan.valid] > 0)
        if reference is None: reference = plan
        else:
            torch.testing.assert_close(plan.admission, reference.admission, rtol=0, atol=0)
            for kind in plan.density: torch.testing.assert_close(plan.density[kind], reference.density[kind], rtol=0, atol=0)
    result = read(model, encoded, query)
    result.square().sum().backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
               for parameter in model.tensor_query_interaction.parameters())
    model.tensor_query_interaction.admission_mode = "curriculum"
    assert model.tensor_query_interaction.build_plan(encoded).sparse_fraction == 1


def test_explicit_soft_constructor_and_invalid_mode():
    from honf_forward_core.interface_fields.tensor_query_interaction import TensorQueryInteractionField
    settings = {"organizer_dim": 16, "global_fast_reader": True, "query_admission_mode": "soft"}
    model = TensorQueryInteractionField(8, 12, 2, 2, architecture="native_context_global_control_honf",
        spatial_dim=2, module_characteristic_length=.1, options=settings)
    model.set_training_progress(epoch=500)
    assert model.tensor_query_interaction.admission_sparse_fraction() == 0
    with pytest.raises(ValueError, match="query_admission_mode"):
        TensorQueryInteractionField(8, 12, 2, 2, architecture="native_context_global_control_honf",
            spatial_dim=2, module_characteristic_length=.1, options={**settings, "query_admission_mode": "unknown"})
