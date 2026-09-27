"""Opt-in intact Run1804 GPU test for the native frozen-cover inverse path.

Set HONF_NATIVE_CHECKPOINT and HONF_NATIVE_STENCIL to run this test with the
local ignored checkpoint and physical atlas. The runner must expose only GPU2
through CUDA_VISIBLE_DEVICES=2.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from channelthermal.data.datasets import GlobalChannelThermalDataset
from channelthermal.evaluation.loading import load_model
from channelthermal.interaction_evidence.reference_adapter import (
    design_state_to_physical_design,
    operating_context_to_named,
)
from channelthermal.interaction_evidence.response_atlas import load_response_atlas_stencil
from channelthermal.inverse.local_interface_contract import AllAccessNativeCoverPolicy, LocalInterfacePlan
from channelthermal.inverse.native_corrected import ThermalNativeQuantityPredictor
from channelthermal.response_control.contracts import DesignInput, role_queries_from_stencil
from channelthermal.response_control.native import DifferentiableThermalOperator
from channelthermal.response_control.runner import _make_input_template, _resolve_dataset_path

from honf_forward_core.interface_fields.adaptive_interaction_cover import MechanismPlan
from honf_inverse_core.contracts import PhysicalDesign


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.skipif(
    not (os.environ.get("HONF_NATIVE_CHECKPOINT") and os.environ.get("HONF_NATIVE_STENCIL")),
    reason="Requires the local ignored Run1804 checkpoint and physical response stencil",
)
def test_intact_thermal_frozen_plan_reaches_all_three_phases_with_live_trial() -> None:
    if not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("Native frozen-cover integration must run on physical GPU2 only.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Native frozen-cover integration requires diagnosed deterministic CUBLAS mode.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    checkpoint_path = Path(os.environ["HONF_NATIVE_CHECKPOINT"]).resolve()
    stencil_path = Path(os.environ["HONF_NATIVE_STENCIL"]).resolve()
    stencil, _ = load_response_atlas_stencil(stencil_path)
    baseline_record = stencil.baseline
    baseline = design_state_to_physical_design(baseline_record.design)
    context = operating_context_to_named(baseline_record.context)
    module_ids = tuple(module.module_id for module in baseline_record.design.modules)
    model, checkpoint = load_model(checkpoint_path, device)
    if int(checkpoint.get("epoch", -1)) != 4738:
        raise ValueError("The integration test requires the reviewed Run1804 e4738 checkpoint.")
    if model.config.core_honf.forward_architecture != "dense_pairwise_field":
        raise ValueError("The integration test requires the intact Run1804 Dense checkpoint.")
    model.eval()
    dataset = GlobalChannelThermalDataset(
        _resolve_dataset_path(checkpoint, None), split="train", points_per_case=1,
        normalize_inputs=False, normalize_targets=False, random_point_sampling=False,
        include_grid=False, include_structure_targets=False,
    )
    operator = DifferentiableThermalOperator(
        model, _make_input_template(dataset),
        dataset_config=checkpoint["train_config"]["dataset"],
        normalization_stats=checkpoint["global_normalization_stats"],
        query_batch_size=2048,
    )
    queries = role_queries_from_stencil(stencil, device=device)
    live = DesignInput.from_state(baseline_record.design, device=device)
    with torch.no_grad():
        dense = operator(live, context.as_mapping(), queries)
    model.core.set_native_interaction_policy(AllAccessNativeCoverPolicy())
    full_plans = operator.capture_anchor_cover_plans(live, context.as_mapping(), queries)
    model.core.set_native_interaction_policy(None)
    assert len(full_plans) == 1 and isinstance(full_plans[0], MechanismPlan)
    with torch.no_grad():
        explicit_full = operator(
            live, context.as_mapping(), queries, fixed_cover_plans=full_plans,
        )
    for role in dense.role_values:
        assert torch.allclose(
            dense.role_values[role], explicit_full.role_values[role], atol=2e-5, rtol=2e-5,
        ), role

    plan = full_plans[0]
    root = plan.tree.nodes[0]
    if root.left is None or root.right is None:
        raise AssertionError("The native receiver tree must expose a real root split.")
    plan = plan.with_split(0, 1.0)
    permission = plan.permission_matrix("QE", phase="P2").clone()
    permission[root.left, -min(32, permission.shape[1]):] = 0.0
    partial = plan.with_permission("QE", permission, phase="P2")
    with torch.no_grad():
        masked_field = operator(live, context.as_mapping(), queries, fixed_cover_plans=(partial,))
    assert float((masked_field.role_values["fluid_fields"] - dense.role_values["fluid_fields"]).abs().max()) > 1e-8
    frozen = LocalInterfacePlan.from_anchor(
        baseline=baseline,
        module_ids_by_slot=module_ids,
        context=context,
        role_queries=queries,
        checkpoint_hash=_sha256(checkpoint_path),
        cover_plans=(partial,),
        max_position_delta=0.11,
    )
    packets = frozen.packet_inventory(module_ids)
    assert {packet.phase for packet in packets} == {"P0", "P1", "P2"}
    assert any(
        packet.mechanism == "QE" and packet.phase == "P2"
        and len(packet.permission_parameters["source_column_indices"]) == plan.environment_count - 32
        for packet in packets
    )
    assert all(
        len(packet.permission_parameters["source_column_indices"]) == plan.environment_count
        for packet in packets if packet.mechanism == "QE" and packet.phase == "P0"
    )
    predictor = ThermalNativeQuantityPredictor(
        operator, queries, device=device, checkpoint_hash=frozen.checkpoint_hash,
        solid_valid_mask=baseline_record.output.roles["solid_temperature"].valid_mask,
    )
    centers = np.array(baseline.module_centers, copy=True)
    centers[0, 0] += 0.05
    trial = PhysicalDesign(
        centers, baseline.module_present.copy(), baseline.heat_powers.copy(),
        module_family_id=baseline.module_family_id,
    )
    with torch.no_grad():
        masked_base = predictor(baseline, context, module_ids, frozen_topology=frozen)
        masked_trial = predictor(trial, context, module_ids, frozen_topology=frozen)
        dense_trial = predictor(trial, context, module_ids, frozen_topology=None)
    assert np.isfinite(masked_base.pressure_drop)
    assert np.isfinite(masked_trial.pressure_drop)
    assert np.isfinite(dense_trial.pressure_drop)
    # Baseline/trial evaluation used the same frozen plan with fresh states.
    assert frozen.cover_plan_hashes == (partial.canonical_hash(),)

    positions = torch.tensor(trial.module_centers, device=device, requires_grad=True)
    heating = torch.tensor(trial.heat_powers, device=device, requires_grad=True)
    caller_owned = DesignInput(
        positions, heating, torch.tensor(trial.module_present > 0.5, device=device),
    )
    quantities = predictor.tensor_quantities_from_input(
        caller_owned, context, module_ids,
        module_family_id=trial.module_family_id, frozen_topology=frozen,
    )
    objective = sum(quantities.module_peak_temperature.values()) + quantities.pressure_drop
    gradients = torch.autograd.grad(objective, (positions, heating))
    assert bool(torch.isfinite(gradients[0]).all()) and bool(torch.isfinite(gradients[1]).all())
    assert float(gradients[0].abs().sum()) > 0.0
    assert float(gradients[1].abs().sum()) > 0.0
