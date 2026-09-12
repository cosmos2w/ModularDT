"""Focused checks for the bounded NStage2 diagnostic entry point."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    _PROJECT_ROOT / "src",
    _PROJECT_ROOT / "Case_ThermalChannel" / "src",
    _PROJECT_ROOT / "tools" / "diagnostics",
):
    sys.path.insert(0, str(_path))

from run_honf_nstage2 import (
    DEFAULT_ANCHOR_CASE_IDS,
    GROUP_MEDIATED_ARCHITECTURE,
    HIERARCHICAL_ARCHITECTURE,
    _a_conditional_route_probe,
    _a_resolution_shell_probe,
    _a_routing_details,
    _b_conditional_route_probe,
    _candidate_modes,
    build_parser,
    group_mediated_zero,
    group_read_zero,
    group_source_zero,
    hierarchical_environment_zero,
    hierarchical_fixed_level_one,
    inventory_existing_artifacts,
    run_interventions,
    run_probes,
    run_smoke,
    run_timing,
)

from honf_forward_core.interface_fields.supports import build_sparse_support_layout


class _HierarchicalBackend:
    def _segmented_read(self, *_args, **_kwargs):
        return torch.full((1, 2, 3), 5.0), {"route": "environment"}

    def read_module(self, *_args, **_kwargs):
        return torch.full((1, 2, 3), 2.0)

    def read(self, *args, **kwargs):
        environment, aux = self._segmented_read(*args, **kwargs)
        return environment + self.read_module(*args, **kwargs), aux


class _ConcreteHierarchicalBackend:
    """Shape of the maintained NStage2-A split boundary."""

    def _segmented_read(self, *_args, **_kwargs):
        return torch.full((1, 2, 3), 5.0), torch.ones((1, 1, 2, 2))

    def read_module(self, *_args, **_kwargs):
        return torch.full((1, 2, 3), 2.0)

    def read(self, *args, **kwargs):
        self.fixed_level = kwargs.get("fixed_level")
        tree, attention = self._segmented_read(*args, **kwargs)
        return tree + self.read_module(*args, **kwargs), {"attention": attention}


class _GroupBackend:
    def read(self, *_args, **_kwargs):
        return torch.full((1, 2, 3), 7.0), {"route": "group"}


class _ProbeHierarchyBackend:
    """Small concrete-shaped backend for routing and conditional probe tests."""

    response_tree_opening_interval = (1.0, 2.0)

    def prepare(self, _encoded, module_states, **_kwargs):
        batch, _modules, hidden = module_states.shape
        tree_states = module_states[:, :1].expand(batch, 3, hidden)
        return {
            "tree_states": tree_states,
            "environment_messages": tree_states,
            "tree_keys": tree_states,
            "tree_values": tree_states,
            "tree_coords": torch.zeros((batch, 3, 2)),
            "tree_mass": torch.ones((batch, 3)),
            "tree_valid": torch.ones((batch, 3), dtype=torch.bool),
            "tree_levels": torch.tensor([2, 1, 0]),
            "tree_bounds_min": torch.tensor([[[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0]]]),
            "tree_bounds_max": torch.tensor([[[1.0, 1.0], [0.5, 0.5], [1.0, 1.0]]]),
        }

    def project_environment_sources(self, tree_states):
        normalized = torch.nn.functional.layer_norm(tree_states, (tree_states.shape[-1],))
        return normalized, normalized

    def _opening_blend(self, receivers, _bounds_min, _bounds_max, _near, _far):
        return torch.ones(receivers.shape[:-1], device=receivers.device)

    def read_environment(self, state, _encoded, receivers, _features, **_kwargs):
        count = int(receivers.shape[1])
        indices = torch.arange(count, device=receivers.device)
        aux = {
            "hierarchical_incidence_batch": torch.zeros(count, dtype=torch.long, device=receivers.device),
            "hierarchical_incidence_query": indices,
            "hierarchical_incidence_node": torch.full_like(indices, 2),
            "hierarchical_incidence_eta": torch.ones(count, device=receivers.device),
            "hierarchical_incidence_attention": torch.ones((count, 2), device=receivers.device),
            "hierarchical_traversal_rows": torch.tensor(float(count), device=receivers.device),
        }
        context = state["tree_values"][:, 2:3].expand(1, count, -1)
        return context, aux

    def read_module(self, state, _encoded, receivers, _features):
        return state["tree_states"][:, :1].expand(1, int(receivers.shape[1]), -1)


def _probe_hierarchy_model() -> tuple[SimpleNamespace, SimpleNamespace]:
    backend = _ProbeHierarchyBackend()
    encoded = SimpleNamespace(module_present=torch.ones(1, 2), module_centers=torch.zeros(1, 2, 2))
    module_states = torch.ones(1, 2, 4)
    prepared = SimpleNamespace(
        backend_state=backend.prepare(encoded, module_states),
        encoded=encoded,
        module_states=module_states,
    )
    model = SimpleNamespace(
        core=SimpleNamespace(backend=backend, _receiver_features=lambda _prepared, receivers: receivers),
        config=SimpleNamespace(
            core_honf=SimpleNamespace(forward_architecture=HIERARCHICAL_ARCHITECTURE)
        ),
    )
    return model, prepared


class _ProbeGroupBackend:
    def __init__(self, cache):
        self.cache = cache

    def prepare(self, _encoded, module_states, layout_cache):
        hidden = int(module_states.shape[-1])
        group_state = module_states[:, :1].expand(1, layout_cache.layout.num_groups, hidden)
        return SimpleNamespace(
            cache=layout_cache,
            group_state=group_state,
            group_keys=group_state,
            group_values=group_state,
        )

    def packed_coarse_group_sources(self, state):
        occupancy = state.cache.layout.occupancy.unsqueeze(0)
        return SimpleNamespace(
            group_states=state.group_state,
            occupancy=occupancy,
            valid=occupancy > 0.0,
        )

    def read(self, state, _encoded, receivers, _features, **_kwargs):
        return state.group_values[:, :1].expand(1, int(receivers.shape[1]), -1), {}


class _ProbeCommon:
    def prepare_coarse(self, _module_states, _env_tokens, _present, _weights, **kwargs):
        return kwargs["packed_group_states"].mean(dim=1, keepdim=True)

    def read_coarse(self, _features, _global_token, coarse_state):
        return coarse_state.expand(1, int(_features.shape[1]), -1)


def _probe_group_model() -> tuple[SimpleNamespace, SimpleNamespace, torch.Tensor]:
    ports = torch.tensor([[[[0.0, 0.0], [0.1, 0.0]], [[2.0, 2.0], [2.1, 2.0]]]])
    present = torch.ones((1, 2))
    env_coords = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
    env_weights = torch.ones((1, 3))
    layout = build_sparse_support_layout(ports, present, env_coords, env_weights, spacing=0.5)
    cache = SimpleNamespace(layout=layout)
    backend = _ProbeGroupBackend(cache)
    common = _ProbeCommon()
    encoded = SimpleNamespace(
        module_centers=torch.tensor([[[0.0, 0.0], [2.0, 2.0]]]),
        global_token=torch.ones((1, 4)),
        env_tokens=torch.ones((1, 3, 4)),
        module_present=present,
        env_weights=env_weights,
    )
    module_states = torch.ones((1, 2, 4))
    prepared = SimpleNamespace(
        backend_state=backend.prepare(encoded, module_states, cache),
        encoded=encoded,
        module_states=module_states,
    )
    model = SimpleNamespace(
        core=SimpleNamespace(
            backend=backend,
            common=common,
            _receiver_features=lambda _prepared, receivers: receivers,
        ),
        config=SimpleNamespace(
            core_honf=SimpleNamespace(forward_architecture=GROUP_MEDIATED_ARCHITECTURE)
        ),
    )
    field_receivers = torch.tensor([[[0.2, 0.2], [0.5, 0.5], [3.0, 3.0]]])
    return model, prepared, field_receivers


def _model(architecture: str, backend: object, common: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        core=SimpleNamespace(
            backend=backend,
            common=common or SimpleNamespace(),
            _interface_read_role="p2_field",
        ),
        config=SimpleNamespace(
            core_honf=SimpleNamespace(forward_architecture=architecture),
        ),
    )


def test_default_anchor_set_is_fixed_before_results() -> None:
    assert DEFAULT_ANCHOR_CASE_IDS == ("0273", "0653", "0283", "0298", "0302")
    assert _candidate_modes(HIERARCHICAL_ARCHITECTURE) == (
        "p0",
        "p1_only",
        "p2",
        "fixed_level1",
    )
    assert _candidate_modes(GROUP_MEDIATED_ARCHITECTURE) == (
        "p0",
        "p1_only",
        "p2",
        "p2_local_only",
        "p2_coarse_group_only",
    )


def test_a_routing_aggregates_receivers_and_compares_zero_eta_node_to_selected_node() -> None:
    model, prepared = _probe_hierarchy_model()
    receivers = {
        "physical_ports": torch.zeros((1, 1, 2)),
        "field_probes": torch.ones((1, 1, 2)),
    }
    details = _a_routing_details(model, prepared, receivers)
    assert details["physical_ports"]["incidence_row_count"] == 1
    assert details["field_probes"]["incidence_row_count"] == 1
    assert details["shared_node_ids"] == [2]
    assert details["zero_weight_direct_influence"]["selected_zero_mass_row_count"] == 0
    probe = details["zero_weight_node_probe"]
    assert probe["status"] == "ok"
    assert probe["nodes"]["omitted_zero_eta"]["jvp_norm"] == 0.0
    assert probe["nodes"]["selected_positive_eta"]["jvp_norm"] > 0.0
    assert probe["state_direction_vector"][:2] == pytest.approx(
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]
    )
    projected, _ = model.core.backend.project_environment_sources(
        prepared.backend_state["tree_states"]
    )
    torch.testing.assert_close(projected.mean(dim=-1), torch.zeros_like(projected.mean(dim=-1)))


def test_a_conditional_probe_separates_fine_em_from_direct_qm() -> None:
    model, prepared = _probe_hierarchy_model()
    result = _a_conditional_route_probe(
        model,
        prepared,
        {"field_probes": torch.zeros((1, 2, 2))},
        0,
    )
    assert result["fineEM_environment_messages"]["nonzero_environment_source_count"] > 0
    receiver_result = result["receivers"]["field_probes"]
    assert "QM_direct_module" in receiver_result
    assert "fineEM" not in receiver_result
    assert result["conditional_ad_fd_validation"]["agreement"] == "reported_without_threshold"
    assert all(
        row["agreement"] == "reported_without_threshold"
        for row in receiver_result["hierarchical_environment"]["steps"].values()
    )


def test_a_resolution_shell_keeps_batched_receiver_shape() -> None:
    model, prepared = _probe_hierarchy_model()
    result = _a_resolution_shell_probe(model, prepared)
    assert result["status"] == "ok"
    assert len(result["endpoints"]) == 5
    assert all("coordinate_read_derivative" in row for row in result["endpoints"])
    assert result["opening_blend_reference"]["float64_reference_only"]
    assert result["opening_blend_reference"]["same_points_bounds_and_interval"]
    assert all(
        "opening_blend_float64_scalar_reference" in row
        and "retained_eta_complement_actual_model_dtype" in row
        for row in result["endpoints"]
    )


def test_b_conditional_probe_uses_farthest_field_receiver_and_reports_selected_influence() -> None:
    model, prepared, field_receivers = _probe_group_model()
    ports = torch.tensor([[[0.0, 0.0], [0.1, 0.0]]])
    result = _b_conditional_route_probe(
        model,
        prepared,
        {"physical_ports": ports, "field_probes": field_receivers},
        0,
    )
    assert result["selected_receiver_indices"]["far_field"] == 2
    assert set(result["selected_receiver_influence"]) == {"near_port", "far_field"}
    assert result["conditional_ad_fd_validation"]["agreement"] == "reported_without_threshold"
    assert all(
        row["agreement"] == "reported_without_threshold"
        for row in result["selected_receiver_influence"]["near_port"][
            "local_group_read"
        ]["steps"].values()
    )
    assert result["selected_receiver_influence"]["near_port"]["detached_group_mediated_coarse"][
        "direct_module_to_coarse_jvp_removed"
    ]


def test_hierarchical_environment_zero_preserves_module_branch_and_scopes_role() -> None:
    model = _model(HIERARCHICAL_ARCHITECTURE, _HierarchicalBackend())
    backend = model.core.backend

    with hierarchical_environment_zero(model, "p2"):
        suppressed, aux = backend.read(None, None, None, None)
        torch.testing.assert_close(suppressed, torch.full((1, 2, 3), 2.0))
        assert aux["route"] == "environment"

    model.core._interface_read_role = "p0_port"
    with hierarchical_environment_zero(model, "p2"):
        normal, _ = backend.read(None, None, None, None)
    torch.testing.assert_close(normal, torch.full((1, 2, 3), 7.0))


def test_hierarchical_environment_zero_patches_concrete_segmented_boundary() -> None:
    model = _model(HIERARCHICAL_ARCHITECTURE, _ConcreteHierarchicalBackend())
    with hierarchical_environment_zero(model, "p2"):
        value, aux = model.core.backend.read(None, None, None, None)
    torch.testing.assert_close(value, torch.full((1, 2, 3), 2.0))
    assert tuple(aux["attention"].shape) == (1, 1, 2, 2)


def test_hierarchical_fixed_level_one_is_p2_read_override() -> None:
    backend = _ConcreteHierarchicalBackend()
    model = _model(HIERARCHICAL_ARCHITECTURE, backend)
    model.core._interface_read_role = "p0_port"
    with hierarchical_fixed_level_one(model):
        backend.read(None, None, None, None)
        assert backend.fixed_level is None
        model.core._interface_read_role = "p2_field"
        backend.read(None, None, None, None)
        assert backend.fixed_level == 1


def test_hierarchical_environment_zero_rejects_missing_split_boundary() -> None:
    model = _model(HIERARCHICAL_ARCHITECTURE, _GroupBackend())
    with pytest.raises(TypeError, match="_segmented_read"), hierarchical_environment_zero(model, "p2"):
        pass


def test_group_read_zero_is_role_scoped() -> None:
    model = _model(GROUP_MEDIATED_ARCHITECTURE, _GroupBackend())
    backend = model.core.backend
    with group_read_zero(model, "p2"):
        suppressed, aux = backend.read(None, None, None, None)
    torch.testing.assert_close(suppressed, torch.zeros((1, 2, 3)))
    assert aux["route"] == "group"

    model.core._interface_read_role = "p1_refinement"
    with group_read_zero(model, "p2"):
        normal, _ = backend.read(None, None, None, None)
    torch.testing.assert_close(normal, torch.full((1, 2, 3), 7.0))


def test_group_source_zero_happens_before_coarse_processor_and_restores_patch() -> None:
    calls: list[bool] = []

    class Common:
        def prepare_coarse(self, *args, group_source_enabled: bool = True, **kwargs):
            del args, kwargs
            calls.append(bool(group_source_enabled))
            return torch.tensor([1.0]) if group_source_enabled else torch.tensor([0.0])

    common = Common()
    model = _model(GROUP_MEDIATED_ARCHITECTURE, _GroupBackend(), common)
    original_function = common.prepare_coarse.__func__
    with group_source_zero(model, "p2"):
        value = common.prepare_coarse(None, None, None, None)
        torch.testing.assert_close(value, torch.tensor([0.0]))
        assert calls == [False]
    assert common.prepare_coarse.__func__ is original_function


def test_group_mediated_zero_stacks_local_and_coarse_removals() -> None:
    calls: list[bool] = []

    class Common:
        def prepare_coarse(self, *args, group_source_enabled: bool = True, **kwargs):
            del args, kwargs
            calls.append(bool(group_source_enabled))
            return torch.tensor([1.0])

    common = Common()
    model = _model(GROUP_MEDIATED_ARCHITECTURE, _GroupBackend(), common)
    with group_mediated_zero(model, "p2"):
        suppressed, _ = model.core.backend.read(None, None, None, None)
        torch.testing.assert_close(suppressed, torch.zeros((1, 2, 3)))
        common.prepare_coarse(None, None, None, None)
    assert calls == [False]


def test_inventory_reports_missing_candidates_without_inventing_best_budget(tmp_path: Path) -> None:
    inventory = inventory_existing_artifacts(tmp_path)
    assert inventory["nstage2_candidates"]["1807"]["directories"] == []
    assert inventory["nstage2_candidates"]["1808"]["directories"] == []
    assert inventory["parent_best_through_epoch_500"]["status"] == "not_located"


def test_interventions_closes_dataset_on_success(monkeypatch, tmp_path: Path) -> None:
    import channelthermal.evaluation.prepared as prepared_module
    import run_honf_nstage2 as entry

    checkpoint_path = tmp_path / "candidate.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    spec = SimpleNamespace(label="candidate", path=checkpoint_path)
    model = SimpleNamespace(
        config=SimpleNamespace(
            core_honf=SimpleNamespace(forward_architecture=HIERARCHICAL_ARCHITECTURE)
        )
    )
    closed = {"value": False}

    class Dataset:
        def close(self) -> None:
            closed["value"] = True

    dataset = Dataset()
    sample = {"case_id": "0273"}

    monkeypatch.setattr(entry, "parse_checkpoint_specs", lambda _values: [spec])
    monkeypatch.setattr(entry, "select_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(entry, "_load_model_spec", lambda _spec, _device: (model, {}))
    monkeypatch.setattr(entry, "_load_dataset", lambda _checkpoint, _args: (dataset, Path("dataset.h5")))
    monkeypatch.setattr(entry, "_load_raw_sample", lambda *_args: {})
    monkeypatch.setattr(prepared_module, "select_sample", lambda *_args: sample)
    monkeypatch.setattr(entry, "_query_points", lambda *_args: np.zeros((2, 2), dtype=np.float32))
    monkeypatch.setattr(
        entry,
        "_run_variant",
        lambda *_args, **_kwargs: {"pred_field": torch.zeros((1, 2, 1))},
    )
    monkeypatch.setattr(entry, "_canonical_ground_truth_errors", lambda *_args: {})
    monkeypatch.setattr(entry, "phase_variant_context", lambda *_args: nullcontext())
    monkeypatch.setattr(entry, "_state_keys", lambda _model: ("weight",))
    monkeypatch.setattr(entry, "_checkpoint_record", lambda *_args: {"label": "candidate"})
    monkeypatch.setattr(entry, "inventory_existing_artifacts", dict)

    args = SimpleNamespace(
        checkpoint=[f"candidate={checkpoint_path}"],
        device="cpu",
        dataset=None,
        split="test",
        case_id=["0273"],
        query_count=2,
    )
    payload = run_interventions(args)
    assert payload["case_ids"] == ["0273"]
    assert closed["value"]


def test_smoke_delegates_to_existing_helper(monkeypatch, tmp_path: Path) -> None:
    import run_regional_response_study

    expected = {"task": "smoke", "delegated": True}
    monkeypatch.setattr(run_regional_response_study, "run_smoke", lambda args: expected)
    args = SimpleNamespace(output=tmp_path / "smoke.json")
    assert run_smoke(args) is expected


def test_parser_supports_phase_intervention_alias() -> None:
    args = build_parser().parse_args(
        [
            "phase-interventions",
            "--checkpoint",
            "candidate=/tmp/candidate.pt",
            "--output",
            "/tmp/nstage2.json",
        ]
    )
    assert args.task == "phase-interventions"
    assert args.case_id is None


def test_parser_supports_plan_104_probes_with_fixed_default_cases() -> None:
    args = build_parser().parse_args(
        [
            "probes",
            "--checkpoint",
            "candidate=/tmp/candidate.pt",
            "--output",
            "/tmp/nstage2-probes.json",
        ]
    )
    assert args.task == "probes"
    assert args.case_id is None
    assert args.query_count == 32


def test_timing_delegates_to_maintained_protocol(monkeypatch, tmp_path: Path) -> None:
    import run_regional_response_study

    expected = {"task": "timing", "delegated": True}
    monkeypatch.setattr(run_regional_response_study, "run_timing_protocol", lambda args: expected)
    args = SimpleNamespace(output=tmp_path / "timing.json")
    assert run_timing(args) is expected


def test_probes_appends_one_result_per_requested_case(monkeypatch, tmp_path: Path) -> None:
    import channelthermal.interface_field_coupling as coupling
    import run_honf_nstage2 as entry

    checkpoint_path = tmp_path / "candidate.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    spec = SimpleNamespace(label="candidate", path=checkpoint_path)
    encoded = SimpleNamespace(
        module_present=torch.ones((1, 2)),
        module_centers=torch.zeros((1, 2, 2)),
    )
    prepared = SimpleNamespace(
        encoded=encoded,
        module_states=torch.ones((1, 2, 4)),
        backend_state=SimpleNamespace(),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            core_honf=SimpleNamespace(
                forward_architecture=HIERARCHICAL_ARCHITECTURE,
            )
        ),
        core=SimpleNamespace(),
    )
    def close_dataset() -> None:
        return None

    dataset = SimpleNamespace(selected_case_ids=["0273", "0298"], close=close_dataset)
    sample = {
        "x_grid": np.asarray([0.0, 1.0], dtype=np.float32),
        "y_grid": np.asarray([0.0, 1.0], dtype=np.float32),
    }

    monkeypatch.setattr(entry, "parse_checkpoint_specs", lambda _values: [spec])
    monkeypatch.setattr(entry, "select_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(entry, "_load_model_spec", lambda _spec, _device: (model, {"epoch": 500}))
    monkeypatch.setattr(entry, "_load_dataset", lambda _checkpoint, _args: (dataset, Path("dataset.h5")))
    monkeypatch.setattr(entry, "select_sample", lambda _dataset, _case_id, _index: sample)
    monkeypatch.setattr(entry, "_query_points", lambda _sample, _count: np.zeros((2, 2), dtype=np.float32))
    monkeypatch.setattr(
        entry,
        "_forward_tensor_batch",
        lambda *_args, **_kwargs: {
            "prepared_state": SimpleNamespace(prepared=prepared),
            "pred_port_condition": torch.zeros((1, 2, 4, 5)),
        },
    )
    monkeypatch.setattr(
        coupling,
        "_port_coordinates",
        lambda _model, centers, ntheta: centers[:, :, None, :].expand(-1, -1, ntheta, -1),
    )
    monkeypatch.setattr(entry, "_physical_coordinate_probe", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(entry, "_a_routing_details", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(entry, "_a_conditional_route_probe", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(entry, "_a_resolution_shell_probe", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(entry, "_state_keys", lambda _model: ("weight",))
    monkeypatch.setattr(entry, "_checkpoint_record", lambda *args, **kwargs: {"label": "candidate"})
    monkeypatch.setattr(entry, "inventory_existing_artifacts", dict)

    args = SimpleNamespace(
        checkpoint=[f"candidate={checkpoint_path}"],
        device="cpu",
        dataset=None,
        split="test",
        case_id=None,
        query_count=2,
    )
    payload = run_probes(args)
    assert payload["case_ids"] == ["0273", "0298"]
    assert [row["case_id"] for row in payload["results"]] == ["0273", "0298"]
