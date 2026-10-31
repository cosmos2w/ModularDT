import numpy as np
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from channelthermal.local_ports import ThermalCalibratedPorts, ThermalPortHead, fit_port_statistics, port_loss
from channelthermal.local_coupling import update_local_module_params_from_ports


def test_calibrated_ports_preserve_input_gradient_and_strict_state():
    ports_builder = ThermalCalibratedPorts(14.8148)
    assert not list(ports_builder.parameters())
    restored = ThermalCalibratedPorts(0.0)
    restored.load_state_dict(ports_builder.state_dict(), strict=True)
    outside = torch.randn(1, 2, 64, requires_grad=True)
    present = torch.tensor([[1.0, 0.0]])
    ports = restored(outside, torch.zeros(1, 2, 3), present)
    torch.testing.assert_close(ports[0, 0, :, 3], outside[0, 0])
    torch.testing.assert_close(ports[0, 0, :, 4], restored.mean[1].expand(64))
    assert ports[0, 1].count_nonzero() == 0
    ports[..., 3].sum().backward()
    torch.testing.assert_close(outside.grad, present[..., None].expand_as(outside))


def test_train_statistics_and_port_masks_exclude_absent_and_invalid_h():
    tokens = np.zeros((2, 2, 5), dtype=np.float32)
    tokens[..., 3] = [[1, 3], [1000, 1000]]
    tokens[..., 4] = [[2, 8], [1000, 1000]]
    cases = [{"teacher_port_tokens": tokens, "structure": {"module_present": [1, 0]},
              "interface_condition_valid_mask": np.array([[1, 0], [1, 1]])}]
    stats = fit_port_statistics(cases)
    assert stats["mean"] == [2, 2]
    assert stats["std"] == [1, 1]
    target = torch.tensor(tokens)[None]
    pred = target.clone().requires_grad_()
    with torch.no_grad():
        pred[..., 3:] += 1
    t, h = port_loss(pred, target, torch.tensor([[1, 0]]), torch.ones(1, 2, 2),
                     torch.tensor([[[1, 0], [1, 1]]]), torch.ones(2))
    assert torch.equal(t, torch.ones(1)) and torch.equal(h, torch.ones(1))
    (t + h).sum().backward()
    assert pred.grad[0, 0, 1, 4] == 0 and pred.grad[0, 1].abs().sum() == 0


def test_port_head_and_local_summaries_preserve_gradient():
    head = ThermalPortHead({"mean": [4, 5], "std": [2, 3]})
    outside = torch.randn(1, 2, 64, requires_grad=True)
    present = torch.tensor([[1., 0.]])
    prescribed = torch.randn(1, 2, 3)
    ports = head(outside, prescribed, present)
    params = update_local_module_params_from_ports(
        torch.cat([prescribed, torch.zeros(1, 2, 4)], dim=-1), ports, present)
    assert ports.shape == (1, 2, 64, 5)
    assert (ports[0, 0, :, 4] >= 0).all() and ports[0, 1].count_nonzero() == 0
    torch.testing.assert_close(params[0, 0, 5], ports[0, 0, :, 3].mean())
    params.sum().backward()
    assert outside.grad[0, 0].abs().sum() > 0


def test_port_case_counts_exclude_empty_channels():
    from channelthermal.training.local_ports_task import ThermalLocalPortsTask
    targets = SimpleNamespace(outside_valid=torch.ones(2, 1, 2, dtype=torch.bool),
                              teacher_ports=torch.ones(2, 1, 2, 5),
                              teacher_h_valid=torch.tensor([[[1, 0]], [[0, 0]]]))
    assert ThermalLocalPortsTask._port_case_counts(targets, torch.ones(2, 1)) == (2, 1)
    provider = object.__new__(ThermalLocalPortsTask)
    provider.model = SimpleNamespace(port_head=SimpleNamespace(std=torch.ones(2)))
    prediction = SimpleNamespace(port_present=torch.ones(2, 1), port_valid=targets.outside_valid,
                                 port_values=torch.zeros(2, 1, 2, 5), work={})
    terms = provider._port_terms(prediction, targets)
    assert terms["port/outside_temperature"].denominator == 2
    assert terms["port/h"].denominator == 1
    assert terms["port/h"].numerator == 1
    targets.outside_valid.zero_()
    assert ThermalLocalPortsTask._port_case_counts(targets, torch.ones(2, 1)) == (0, 0)
    terms = provider._port_terms(prediction, targets)
    assert terms["port/h"].denominator == 0 and terms["port/h"].numerator == 0


def test_oracle_requires_all_ports_valid_and_finite():
    from channelthermal.local_ports_evaluation import _oracle_module_valid
    present = torch.tensor([[1, 1, 1, 1, 0]])
    outside = torch.ones(1, 5, 64, dtype=torch.bool)
    h_valid = torch.ones_like(outside)
    teacher = torch.ones(1, 5, 64, 5)
    outside[0, 1, 3] = False
    h_valid[0, 2, 8] = False
    teacher[0, 3, 5, 3] = float("nan")
    assert _oracle_module_valid(present, outside, h_valid, teacher).tolist() == [[True, False, False, False, False]]
    h_valid.zero_()
    assert not _oracle_module_valid(present, outside, h_valid, teacher).any()


def test_actual_native_port_union_matches_full_fluid_union():
    from channelthermal.local_ports import organized_outside64
    from channelthermal.native_observations import interpolate_native
    from channelthermal.training.local_ports_task import build_thermal_local_ports_task
    from honf_runtime.unified_training import SamplingKey

    root = Path(__file__).resolve().parents[2]
    recipe = json.loads((root / "configs/research/thermal_r15_2_localports_train150_v1.json").read_text())
    if not Path(recipe["dataset"]["data_path"]).exists():
        pytest.skip("Thermal native dataset unavailable")
    recipe["training"]["microbatch_cases"] = 1
    recipe["member_routing"]["health"]["enabled"] = False
    model, provider = build_thermal_local_ports_task(recipe, device="cpu")
    key = SamplingKey(0, 1, 0, 0, "joint", model.architecture,
                      sampling_version=provider.sampling_version,
                      dataset_id=provider.sampling_dataset_id)
    batch = provider.make_batch(provider.epoch_cases(1, 0)[:1], key)
    model.eval()
    with torch.no_grad():
        prepared = model.prepare_native(batch.scene_inputs.structure, batch.receivers.fluid_xy,
                                        physical_heat=batch.receivers.heat,
                                        local_query_points=batch.receivers.local_query_points,
                                        ntheta=16, scene_key=batch.scene_inputs.case_ids)
        actual, actual_valid, work = organized_outside64(model, prepared)
        assert torch.equal(actual_valid, batch.targets.outside_valid)
        query, union, stencils, *rest = model._query_union(prepared.scene_batch, prepared.fluid_xy, None, 64)
        output = model.core.decode(prepared.organization, query, intervention={"read_mode": "restricted"})
        temperature = model._physical_temperature(output.joint_fields[..., 3])
        reference = interpolate_native(temperature[..., None], stencils["outside"]).reshape_as(actual)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert work["native_union_rows"] < int(union.unique_counts.sum())

    from channelthermal.local_ports_evaluation import evaluate_direct_native, evaluate_local_port_readouts
    checkpoint = root / recipe["local_ports"]["checkpoint_path"]
    if not checkpoint.exists():
        pytest.skip("Frozen local checkpoint unavailable")
    review = evaluate_local_port_readouts(model, provider, batch)
    assert set(review["local"]) == {"predicted", "oracle", "train_mean_h"}
    torch.testing.assert_close(review["ports"]["train_mean_h"][..., 3], actual)
    assert review["metrics"]["ports"]["outside_temperature"]["count"] > 0
    assert set(review["metrics"]["direct_fluid"]) == {"u", "v", "pressure", "vorticity", "temperature"}
    assert review["metrics"]["direct_fluid"]["temperature"]["count"] > 0
    assert review["metrics"]["oracle"]["solid_temperature"]["count"] > 0
    matched = review["metrics"]["oracle_eligible_comparison"]
    for role in ("solid_temperature", "surface_temperature", "q_proxy", "module_peak_sampled_points"):
        counts = {entry[role]["count"] for entry in matched.values()}
        assert len(counts) == 1 and next(iter(counts)) > 0
    class OneNonfinite(torch.nn.Module):
        def __init__(self, local):
            super().__init__()
            self.local = local
            self.calls = 0

        def forward(self, *args):
            output = self.local(*args)
            self.calls += 1
            if self.calls == 1:
                output = dict(output)
                output["internal_temperature"] = output["internal_temperature"].clone()
                output["internal_temperature"][0, 0, 0, 0] = float("nan")
            return output

    nonfinite = evaluate_local_port_readouts(model, provider, batch, local_readout=OneNonfinite(provider.build_local_readout()))
    counts = [value["solid_temperature"]["count"] for value in nonfinite["metrics"]["oracle_eligible_comparison"].values()]
    assert len(set(counts)) == 1 and counts[0] == matched["direct"]["solid_temperature"]["count"] - 1
    assert review["timing_seconds"]["complete_predicted"] >= review["timing_seconds"]["direct_decode"]
    invalid_batch = replace(batch, targets=replace(batch.targets, teacher_h_valid=torch.zeros_like(batch.targets.teacher_h_valid)))
    invalid_review = evaluate_local_port_readouts(model, provider, invalid_batch)
    assert invalid_review["metrics"]["oracle"]["solid_temperature"]["count"] == 0
    assert invalid_review["metrics"]["predicted"]["solid_temperature"]["count"] > 0
    assert invalid_review["oracle_conditioning"]["valid_modules"] == 0
    for entry in invalid_review["metrics"]["oracle_eligible_comparison"].values():
        assert all(entry[role]["count"] == 0 and entry[role]["rmse"] is None
                   for role in ("solid_temperature", "surface_temperature", "q_proxy", "module_peak_sampled_points"))
    full_batch = next(provider.native_evaluation_batches(
        "validation", case_keys=[provider.validation_case_ids[0]], microbatch_size=1))
    full_review = evaluate_local_port_readouts(model, provider, full_batch)
    control_review = evaluate_direct_native(model, full_batch)
    torch.testing.assert_close(control_review["direct"]["pred_field"], full_review["direct"]["pred_field"], rtol=0, atol=0)
    assert control_review["metrics"]["direct"] == {key: full_review["metrics"]["direct"][key] for key in control_review["metrics"]["direct"]}
    assert full_batch.receivers.ntheta == 64
    assert full_review["metrics"]["oracle"]["surface_temperature"]["count"] > 0
    assert full_review["metrics"]["direct_fluid"]["u"]["count"] > 0
    assert full_review["metrics"]["oracle"]["solid_temperature"]["count"] > review["metrics"]["oracle"]["solid_temperature"]["count"]
