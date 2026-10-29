"""Public model names retain the concrete models' original checkpoint contracts."""
from copy import deepcopy

import pytest
import torch
from torch.nn.parameter import is_lazy

from honf_forward_core.config import InterfaceFieldConfig, UnifiedForwardConfig
from honf_forward_core.content_hypergraph import ContentHypergraphCoreV2
from honf_forward_core.continuous_joint.unified import UnifiedJointCore
from honf_forward_core.interaction_reused_member_routing.anchored import AnchoredMemberRoutingCore
from honf_forward_core.interface_fields.core import InterfaceFieldCore
from honf_forward_core.model import HONFNeuralField
from honf_forward_core.models import MODEL_FAMILIES, build_model, model_family


def _content_config():
    return ContentHypergraphCoreV2(
        module_input_dim=9, environment_input_dim=6, operating_input_dim=12,
        coordinate_dim=2, query_input_dim=0, output_dim=4, architecture="A2-D",
        context_dim=16, num_heads=4, context_depth=2, local_hidden_dim=16,
        dropout=0., init_seed=731,
    ).get_config()


def _legacy_config(architecture):
    return UnifiedForwardConfig(
        field_dim=3, domain_length_x=12., domain_length_y=6.,
        num_env_tokens_x=4, num_env_tokens_y=2, num_hyperedges=3,
        hidden_dim=24, dropout=0., forward_architecture=architecture,
        interface_model=(InterfaceFieldConfig(message_hidden_dim=12, attention_heads=4,
                                              coarse_latent_count=4, coarse_blocks=1)
                         if architecture == "dense_pairwise_field" else None),
    )


def _reference(name):
    base = _content_config()
    if name == "hypergraph_routed":
        config = _legacy_config("legacy_honf")
        return HONFNeuralField(config), config.to_dict()
    if name == "dense_pairwise":
        config = _legacy_config("dense_pairwise_field")
        return InterfaceFieldCore(config), config.to_dict()
    if name == "content_attention":
        return ContentHypergraphCoreV2.from_config(base), base
    if name == "unified_hypergraph":
        model = UnifiedJointCore(base, input_dataset="thermal", routing_mode="hypergraph")
        return model, model.get_config()
    access = "active_domain" if name == "relation_anchored_active" else "inventory_unit_total"
    model = AnchoredMemberRoutingCore(base, input_dataset="thermal", access_measure=access,
                                     ee_incidence_aggregation="post_gate",
                                     ee_gate_mode="relative_quadratic", ee_initial_tau=1.25)
    return model, model.get_config()


@pytest.mark.parametrize("family", [family.name for family in MODEL_FAMILIES])
def test_factory_preserves_original_config_and_state_inventory(family):
    original, config = _reference(family)
    saved = deepcopy(config)
    rebuilt = build_model(family, config)
    assert type(rebuilt) is type(original)
    assert config == saved
    actual_config = rebuilt.get_config() if hasattr(rebuilt, "get_config") else rebuilt.config.to_dict()
    assert actual_config == saved
    original_state, rebuilt_state = original.state_dict(), rebuilt.state_dict()
    assert list(original_state) == list(rebuilt_state)
    def inventory(state):
        return [(key, (None if is_lazy(value) else value.shape, value.dtype)
                 if torch.is_tensor(value) else value) for key, value in state.items()]
    assert inventory(original_state) == inventory(rebuilt_state)
    assert list(dict(original.named_parameters())) == list(dict(rebuilt.named_parameters()))


def test_documented_aliases_and_unknown_names():
    assert len(MODEL_FAMILIES) == 6
    for family in MODEL_FAMILIES:
        for alias in (family.name, *family.development_names):
            assert model_family(alias.upper()) == family
    with pytest.raises(ValueError, match="Unknown model"):
        model_family("unrecognized")


@pytest.mark.parametrize("name,wrong", [("relation_anchored", "active_domain"),
                                      ("relation_anchored_active", "inventory_unit_total")])
def test_relation_family_rejects_other_access_law(name, wrong):
    _, config = _reference(name)
    config["access_measure"] = wrong
    saved = deepcopy(config)
    with pytest.raises(ValueError, match="access configuration"):
        build_model(name, config)
    assert config == saved


def test_factory_rejects_wrong_legacy_family_and_unknown_config():
    config = _legacy_config("dense_pairwise_field").to_dict()
    with pytest.raises(ValueError, match="forward_architecture"):
        build_model("hypergraph_routed", config)
    config["unrecognized_field"] = 1
    with pytest.raises(ValueError, match="Unknown"):
        build_model("dense_pairwise", config)


def test_unified_factory_rejects_wrong_schema_or_routing():
    _, config = _reference("unified_hypergraph")
    bad = deepcopy(config)
    bad["routing_mode"] = "dense"
    with pytest.raises(ValueError, match="routing_mode"):
        build_model("unified_hypergraph", bad)
    bad = deepcopy(config)
    bad["schema"] = "wrong"
    with pytest.raises(ValueError, match="inventory"):
        build_model("unified_hypergraph", bad)


def test_unified_factory_requires_the_saved_named_input_contract():
    _, config = _reference("unified_hypergraph")
    config.pop("input_contract_sha256")
    with pytest.raises(ValueError, match="input capability inventory"):
        build_model("unified_hypergraph", config)


@pytest.mark.parametrize("family", MODEL_FAMILIES, ids=lambda family: family.name)
def test_development_alias_builds_same_family(family):
    original, config = _reference(family.name)
    rebuilt = build_model(family.development_names[0].upper(), config)
    assert type(rebuilt) is type(original)


@pytest.mark.parametrize("architecture,name", [("legacy_honf", "hypergraph_routed"),
                                                  ("dense_pairwise_field", "dense_pairwise")])
def test_legacy_dataclass_input_is_accepted_without_mutation(architecture, name):
    config = _legacy_config(architecture)
    saved = deepcopy(config.to_dict())
    model = build_model(name, config)
    assert model.config.to_dict() == saved
    assert config.to_dict() == saved


@pytest.mark.parametrize("workflow", ["joint_native_train_v1", "native_retrieval_joint_v1",
                                       "native_budgeted_dispatch_joint_v2", "moment_hyperedges_child_v1",
                                       "inherited_attention_joint_v1"])
def test_content_loader_rejects_retired_workflows(workflow, monkeypatch):
    from honf_runtime import content_hypergraph
    monkeypatch.setattr(content_hypergraph, "load_trusted_checkpoint", lambda *a, **kw: {"workflow": workflow})
    with pytest.raises(ValueError, match="content-hypergraph joint family"):
        content_hypergraph.load_content_hypergraph_checkpoint("unused.pt")
