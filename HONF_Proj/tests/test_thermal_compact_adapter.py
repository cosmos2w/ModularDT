"""Compact control groups retain native Thermal extraction and physical sources."""
import pytest
import torch

from channelthermal.content_hypergraph import ThermalContentHypergraphAdapterV2
from honf_forward_core.continuous_joint.unified import UnifiedJointCore
from tests.test_thermal_content_hypergraph_v2 import _core_config, _receivers, _stats, _structure


def _compact_adapter(modules):
    from honf_forward_core.continuous_joint.unified import CompactUnifiedJointCore

    parent = UnifiedJointCore({**_core_config("A2-D"), "context_depth": 2}, input_dataset="thermal",
                              routing_mode="hypergraph", native_backend="dense_weighted",
                              checkpoint_pipeline=False)
    core = CompactUnifiedJointCore.from_unified_parent(parent)
    model = ThermalContentHypergraphAdapterV2(_stats(), core=core, nx=8, ny=6)
    structure = _structure()
    centers = torch.tensor([[x, y] for x in (1.5, 4.5, 7.5, 10.5)
                            for y in (1., 3., 5.)])[:modules][None]
    structure.update(module_centers=centers, module_present=torch.ones(1, modules),
                     module_source_ids=torch.arange(31, 31 + modules)[None])
    fluid, _ = _receivers()
    local = torch.tensor([[0., 0.], [.35, 0.]]).expand(1, modules, 2, 2).clone()
    return model, structure, fluid, local


@pytest.mark.parametrize("modules", [1, 12])
def test_compact_native_roles_keep_zero_heat_superposition_and_original_columns(modules):
    model, structure, fluid, local = _compact_adapter(modules)
    model.eval()
    prepared = model.prepare_native(structure, fluid, local_query_points=local,
                                    ntheta=16, scene_key=("compact-native-law",))
    heat = torch.linspace(.2, .8, modules)[None].requires_grad_()
    zero = model.apply_native(prepared, torch.zeros_like(heat))
    prediction = model.apply_native(prepared, heat)
    doubled = model.apply_native(prepared, 2 * heat)
    assert zero["pred_field"][..., 4].count_nonzero() == 0
    assert zero["pred_interface"].count_nonzero() == 0
    assert zero["pred_internal_temperature"].count_nonzero() == 0
    torch.testing.assert_close(prediction["pred_field"][..., :4],
                               zero["pred_field"][..., :4], rtol=0, atol=0)
    for name in ("fluid_temperature", "pred_interface", "pred_internal_temperature"):
        torch.testing.assert_close(doubled[name], 2 * prediction[name], rtol=3e-6, atol=3e-6)
    assert prepared.organization.scene.num_modules == modules
    # Each source keeps its own physical column even though retrieval uses K.
    raw = model.core.decode(prepared.organization, prepared.query_batch, controls=heat)
    assert raw.thermal_temperature.shape == prepared.query_batch.coordinates.shape[:2]
    gradient = torch.autograd.grad(prediction["fluid_temperature"].square().sum(), heat)[0]
    assert gradient.shape == (1, modules)
    assert torch.isfinite(gradient).all() and gradient.count_nonzero() > 0


def test_compact_adapter_loads_from_own_payload_with_no_parent_or_dataset():
    model, structure, fluid, local = _compact_adapter(2)
    model.eval()
    payload = model.state_payload()
    assert payload["core"]["schema"] == "compact_unified_joint_v1"
    restored = ThermalContentHypergraphAdapterV2.from_state_payload(
        payload, state_dict=model.state_dict(), device="cpu").eval()
    assert restored.normalizer_key == model.normalizer_key
    assert restored.get_config() == model.get_config()
    predictions = []
    for adapter in (model, restored):
        prepared = adapter.prepare_native(structure, fluid, local_query_points=local,
                                          ntheta=16, scene_key=("self-contained",))
        predictions.append(adapter.apply_native(prepared, torch.tensor([[.3, .7]])))
    for name in ("pred_field", "pred_interface", "pred_internal_temperature"):
        torch.testing.assert_close(predictions[0][name], predictions[1][name], rtol=0, atol=0)


def test_native_backward_trains_compact_controls_and_shared_sibling_policy():
    model, structure, fluid, local = _compact_adapter(12)
    core = model.core
    core.set_compact_phase(30, arm="merge")
    core.set_compact_loss_coefficients(tie=.01, capacity=.01)
    core.joint_begin_update(4030, 0)
    prepared = model.prepare_native(structure, fluid, local_query_points=local,
                                    ntheta=16, scene_key=("compact-training",))
    prediction = model.apply_native(prepared, torch.ones(1, 12))
    scene_losses = core.compact_structural_loss_components(prepared.organization)
    objective = (prediction["pred_field"].square().mean()
                 + prediction["pred_interface"].square().mean()
                 + prediction["pred_internal_temperature"].square().mean()
                 + .01 * scene_losses["tie"].mean() + .01 * scene_losses["capacity"].mean())
    objective.backward()
    for name in ("controls", "route_scorers", "query_scorers", "compatibility"):
        parameters = getattr(core.compact_retrieval, name).parameters()
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.count_nonzero() > 0 for gradient in gradients), name


def test_compact_execution_omits_retired_directory_and_native_seed_scorer(monkeypatch):
    import honf_forward_core.continuous_joint.model as continuous
    import honf_forward_core.continuous_joint.unified as unified

    model, structure, fluid, local = _compact_adapter(12)
    core = model.core
    # A software transition fixture; scientific runs use the learned b > 0 cut.
    with torch.no_grad():
        for policy in core.compact_retrieval.compatibility.values():
            policy.output.bias.fill_(2.)
    core.set_compact_phase(40, arm="merge")
    core.activate_hard_cut()
    model.eval()

    def retired(*args, **kwargs):
        raise AssertionError("Retired native-seed retrieval executed")

    monkeypatch.setattr(unified, "applied_native_directory", retired)
    monkeypatch.setattr(core.router, "forward", retired)
    original_score = continuous.raw_scores

    def information_e_only(reader, *args, **kwargs):
        assert reader is core.environment_reader
        return original_score(reader, *args, **kwargs)

    monkeypatch.setattr(continuous, "raw_scores", information_e_only)
    prepared = model.prepare_native(structure, fluid, local_query_points=local,
                                    ntheta=16, scene_key=("direct-compact-call",))
    compact = prepared.organization.compact_groups
    assert compact.membership.shape == (1, 102, 204)
    assert compact.allocated_count == 102
    prediction = model.apply_native(prepared, torch.ones(1, 12))
    assert torch.isfinite(prediction["pred_field"]).all()
    assert not hasattr(prepared.organization, "continuous_incidence")
