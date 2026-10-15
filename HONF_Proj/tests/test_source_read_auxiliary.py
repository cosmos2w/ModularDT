from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

import honf_runtime.source_read_auxiliary as auxiliary_module
from honf_forward_core.content_hypergraph.source_read_plans import SourceReadControllerV1
from honf_runtime.source_read_auxiliary import (
    STATE_SCHEMA,
    SourceReadAuxiliaryManagerV1,
    _TeacherCase,
    _calibrate_thresholds,
    _context_scales,
    _factory_physical_loss,
    _normalize_teacher_case,
    fit_source_read_child_auxiliary,
)
from honf_runtime.source_read_child import (
    _state_digest,
    _validate_train_fidelity_screen,
    run_source_read_child_auxiliary_fit,
)
from honf_runtime.unified_training import SamplingKey
from honf_runtime.unified_training import named_optimizer_state


class _DirectReader(nn.Module):
    def __init__(self, *, completion: bool):
        super().__init__()
        self.message_output = nn.Linear(8, 8)
        self.thermal_output = nn.Linear(8, 1)
        self.source_read = SourceReadControllerV1(
            width=8,
            coordinate_dim=1,
            heads=2,
            thermal=True,
            rank=3,
            scorer_width=6,
            completion=completion,
        )
        self.source_read.register_buffer("primary_operating_point", torch.zeros((), dtype=torch.long))


class _Core(nn.Module):
    def __init__(self, *, completion: bool):
        super().__init__()
        self.physical = nn.Linear(1, 1)
        self.direct_reader = _DirectReader(completion=completion)


class _Model(nn.Module):
    def __init__(self, *, variant: str, seed: int = 18):
        super().__init__()
        torch.manual_seed(seed)
        self.core = _Core(completion=variant == "S-Complete")


class _Provider:
    seed = 0
    sampling_dataset_id = "source-read-fixture-train"
    sampling_version = SamplingKey.CASE_EPOCH_VERSION

    def __init__(self):
        self.trace_calls: list[tuple[int, int, int]] = []
        self.envelope_calls = 0
        self.training_queries = 0
        self.calibration_state = 0

    def training_state_dict(self):
        return {"fixture": self.calibration_state}

    def source_read_positive_train_envelope(self, train_case_keys):
        self.envelope_calls += 1
        assert tuple(train_case_keys) == (11, 12, 13)
        return 2.5


def _trace_factory(provider, model, case_key, key, max_query_rows):
    provider.trace_calls.append((int(case_key), key.epoch, max_query_rows))
    generator = torch.Generator(device="cpu").manual_seed(1000 + int(case_key) + 17 * key.epoch)
    queries, sources, width, heads = 5, 3, 8, 2
    query_states = torch.randn(1, queries, width, generator=generator)
    module_states = torch.randn(1, sources, width, generator=generator)
    descriptor = torch.randn(1, queries, sources, 3, generator=generator)
    alpha = torch.softmax(torch.randn(1, queries, sources, heads, generator=generator), dim=2)
    hidden = torch.randn(1, queries, sources, width, generator=generator)
    reader = model.core.direct_reader
    with torch.no_grad():
        fine_messages = reader.message_output(hidden)
        thermal_k = reader.thermal_output(fine_messages).squeeze(-1)
    source_present = torch.tensor([[True, True, case_key != 12]])
    query_mask = torch.tensor([[True, True, True, True, case_key != 13]])
    valid_sources = source_present[:, None, :].expand(1, queries, sources)
    return {
        "query_states": query_states,
        "module_states": module_states,
        "descriptor": descriptor,
        "alpha": alpha,
        "fine_hidden": hidden,
        "fine_messages": fine_messages,
        "thermal_k": thermal_k,
        "query_mask": query_mask,
        "source_present_mask": source_present,
        "local_radius": torch.tensor([[0.75, 1.0, 1.25]]),
        "distance_to_sources": torch.rand(1, queries, sources, generator=generator) * 1.4,
        "fine_executed_rows": queries * sources,
        "fine_present_rows": int((query_mask[:, :, None] & valid_sources).sum().item()),
    }


def _physical_loss_factory(provider, model, case_key, key):
    del provider, case_key, key
    return model.core.physical(torch.tensor([[0.7]])).square().mean()


def _added_names(model):
    return sorted(name for name, _ in model.named_parameters() if "direct_reader.source_read." in name)


def _attach(variant="S-Utility", *, max_updates=4):
    model = _Model(variant=variant)
    provider = _Provider()
    inherited = {name: parameter for name, parameter in model.named_parameters()
                 if "direct_reader.source_read." not in name}
    added = {name: parameter for name, parameter in model.named_parameters()
             if "direct_reader.source_read." in name}
    for parameter in inherited.values():
        parameter.requires_grad_(False)
    for parameter in added.values():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    old_values = {name: value.detach().clone() for name, value in model.state_dict().items()
                  if name not in {key for key in model.state_dict()
                                  if "direct_reader.source_read." in key}}
    receipt = fit_source_read_child_auxiliary(
        model,
        provider,
        optimizer,
        added_parameter_names=_added_names(model),
        train_case_keys=(11, 12, 13),
        max_updates=max_updates,
        parent_sha256="a" * 64,
        source_epoch=500,
        teacher_trace_factory=_trace_factory,
        physical_loss_factory=_physical_loss_factory,
    )
    return model, provider, optimizer, receipt, old_values


@pytest.mark.parametrize("variant", ["S-Utility", "S-Complete"])
def test_initial_fit_is_train_only_bounded_and_preserves_inherited_parameters(variant):
    model, provider, optimizer, receipt, old_values = _attach(variant)
    assert receipt["schema"] == "source_read_child_auxiliary_fit_v1"
    assert receipt["partition"] == "TRAIN"
    assert receipt["source_epoch"] == 500
    assert receipt["source_checkpoint_sha256"] == "a" * 64
    assert receipt["train_case_keys"] == [11, 12, 13]
    assert receipt["auxiliary_fit_case_keys"] == [11, 12, 13]
    assert receipt["auxiliary_fit_case_share"] == 1.0
    assert receipt["teacher_rows"] > 0
    assert receipt["teacher_present_rows"] > 0
    assert receipt["teacher_rows"] == 45
    assert receipt["teacher_present_rows"] == 37
    assert receipt["optimizer_updates"] == 4
    assert receipt["updated_parameter_names"] == _added_names(model)
    assert set(receipt["optimizer_steps_by_parameter"].values()) == {4}
    assert receipt["optimizer_algorithm"]["name"] == "AdamW"
    assert receipt["optimizer_algorithm"]["gradient_clip_norm"] == 1.0
    calibration = receipt["gradient_calibration"]
    assert calibration["schema"] == "source_read_gradient_calibration_v1"
    assert calibration["partition"] == "TRAIN"
    assert calibration["physical_all_inherited_parameter_gradient_l2_norm"] > 0
    assert calibration["auxiliary_all_new_parameter_gradient_l2_norm"] > 0
    assert calibration["fixed_auxiliary_coefficient"] == pytest.approx(min(
        1.0,
        0.05 * calibration["physical_all_inherited_parameter_gradient_l2_norm"]
        / calibration["auxiliary_all_new_parameter_gradient_l2_norm"],
    ))
    assert calibration["realized_weighted_auxiliary_to_physical_gradient_share"] <= 0.05 + 1e-12
    screen = receipt["train_fidelity_screen"]
    assert screen["partition"] == "TRAIN"
    assert screen["target_union_retention"] == [0.75, 0.5]
    assert len(screen["actual_union_retention"]) == 2
    assert len(screen["flow_direct_context_relative_rms"]) == 2
    assert len(screen["thermal_positive_H_endpoint_relative_rms"]) == 2
    assert screen["primary_operating_point"] == int(
        model.core.direct_reader.source_read.primary_operating_point
    )
    assert receipt["changed_buffer_names"] == sorted(receipt["changed_buffer_names"])
    assert receipt["changed_buffer_names"]
    assert provider.envelope_calls == 1
    assert {max_rows for _, _, max_rows in provider.trace_calls} == {256}
    assert all(epoch == 500 for _, epoch, _ in provider.trace_calls)
    assert getattr(provider, "source_read_auxiliary_manager") is not None
    manager = provider.source_read_auxiliary_manager
    assert isinstance(manager, SourceReadAuxiliaryManagerV1)
    assert manager.auxiliary_coefficient is not None and 0.0 <= manager.auxiliary_coefficient <= 1.0
    assert manager.gradient_calibration == calibration
    assert manager.auxiliary_coefficient == calibration["fixed_auxiliary_coefficient"]
    controller = model.core.direct_reader.source_read
    assert bool(controller.calibrated)
    assert bool(controller.trained)
    assert bool(controller.metadata_frozen)
    assert (controller.thresholds[0] <= controller.thresholds[1]).item()
    assert torch.all(controller.context_scales > 0)
    assert controller.heating_envelope.item() == pytest.approx(2.5)
    for name, value in old_values.items():
        assert torch.equal(model.state_dict()[name], value), name
    for name, parameter in model.named_parameters():
        if "direct_reader.source_read." not in name:
            assert parameter.grad is None
            assert parameter not in optimizer.state


def test_auxiliary_loss_and_eight_update_refresh_use_one_bounded_train_case():
    model, provider, _, _, _ = _attach("S-Utility")
    manager = provider.source_read_auxiliary_manager
    refresh_before = len(provider.trace_calls)
    for update in range(1, 8):
        result = manager.before_update(
            model, provider,
            absolute_epoch=501,
            ordinary_update_index=update,
            primary_fine_rows=100_000,
        )
        assert result["refreshed"] is False
    result = manager.before_update(
        model, provider,
        absolute_epoch=501,
        ordinary_update_index=8,
        primary_fine_rows=100_000,
    )
    assert result["refreshed"] is True
    assert result["case_key"] in (11, 12, 13)
    assert result["teacher_rows"] <= 128 * 3
    assert result["teacher_row_share"] < 0.05
    assert len(provider.trace_calls) == refresh_before + 1
    assert provider.trace_calls[-1][1:] == (501, 128)
    loss = manager.auxiliary_loss(model)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for name, parameter in model.named_parameters()
               if "direct_reader.source_read." in name)
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if "direct_reader.source_read." not in name)

    saved = manager.state_dict()
    assert saved["schema"] == STATE_SCHEMA
    restored = SourceReadAuxiliaryManagerV1(
        model.core.direct_reader.source_read,
        variant="S-Utility",
        train_case_keys=(11, 12, 13),
        seed=0,
        sampling_dataset_id="source-read-fixture-train",
    )
    restored.load_state_dict(copy.deepcopy(saved))
    assert restored.coadaptation_teacher_rows == manager.coadaptation_teacher_rows
    assert restored.coadaptation_teacher_present_rows == manager.coadaptation_teacher_present_rows
    assert restored.last_ordinary_update_index == 8
    assert restored.refresh_case_key == manager.refresh_case_key
    assert torch.equal(restored.auxiliary_loss(model), manager.auxiliary_loss(model))


def test_refresh_preflight_skips_when_five_percent_budget_cannot_cover_dense_teacher():
    model, provider, _, _, _ = _attach("S-Utility")
    manager = provider.source_read_auxiliary_manager
    calls_before = len(provider.trace_calls)
    result = manager.before_update(
        model, provider,
        absolute_epoch=501,
        ordinary_update_index=8,
        primary_fine_rows=100,
    )
    assert result == {"refreshed": False, "reason": "five_percent_teacher_row_budget"}
    assert len(provider.trace_calls) == calls_before


def test_wrong_variant_and_wrong_resume_cohort_are_rejected():
    controller = SourceReadControllerV1(
        width=8, coordinate_dim=1, heads=2, thermal=True, completion=False
    )
    with pytest.raises(ValueError, match="variant and controller"):
        SourceReadAuxiliaryManagerV1(
            controller,
            variant="S-Complete",
            train_case_keys=(1,),
            seed=0,
            sampling_dataset_id="fixture",
        )
    model, provider, _, _, _ = _attach("S-Utility")
    manager = provider.source_read_auxiliary_manager
    state = manager.state_dict()
    bad_state = copy.deepcopy(state)
    bad_state["train_case_keys"] = [11, 12, 14]
    with pytest.raises(ValueError, match="different child or TRAIN cohort"):
        manager.load_state_dict(bad_state)
    bad_gradient = copy.deepcopy(state)
    bad_gradient["gradient_calibration"]["realized_weighted_auxiliary_to_physical_gradient_share"] = 0.051
    with pytest.raises(ValueError, match="fixed TRAIN rule"):
        manager.load_state_dict(bad_gradient)
    assert model.core.direct_reader.source_read.completion_enabled is False


def test_child_attachment_wrapper_accepts_exact_auxiliary_receipt_and_new_buffer_changes():
    model = _Model(variant="S-Utility", seed=719)
    provider = _Provider()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    optimizer.param_groups[0]["group_name"] = "fixture_joint"
    # Create genuine inherited moments only; the source-read parameters are
    # newly born and must have no optimizer age before the callback.
    optimizer.zero_grad(set_to_none=True)
    model.core.physical(torch.tensor([[0.4], [0.8]])).square().mean().backward()
    optimizer.step()
    names = dict(model.named_parameters())
    added_names = _added_names(model)
    added_set = set(added_names)
    old_state = {name: value.detach().clone() for name, value in model.state_dict().items()
                 if "direct_reader.source_read." not in name}
    parent_optimizer = named_optimizer_state(model, optimizer)
    new_buffers = {name: value for name, value in model.named_buffers()
                   if "direct_reader.source_read." in name}
    declaration = {
        "auxiliary_fit_mode": "calibrated",
        "auxiliary_update_limit": 4,
        "source_checkpoint_sha256": "c" * 64,
        "new_source_read_buffer_names": sorted(new_buffers),
        "new_source_read_buffer_initial_sha256": _state_digest(new_buffers),
    }
    callback = lambda *args, **kwargs: fit_source_read_child_auxiliary(
        *args,
        **kwargs,
        teacher_trace_factory=_trace_factory,
        physical_loss_factory=_physical_loss_factory,
    )
    receipt = run_source_read_child_auxiliary_fit(
        callback,
        model,
        provider,
        optimizer,
        {
            "model_state_dict": old_state,
            "optimizer_state_by_name": parent_optimizer,
            "provider_training_state": provider.training_state_dict(),
        },
        declaration,
        added_parameter_names=added_names,
        train_case_keys=(11, 12, 13),
    )
    assert receipt["mode"] == "calibrated"
    assert receipt["optimizer_updates"] == 4
    assert receipt["changed_buffer_names"]
    assert set(receipt["updated_parameter_names"]) == added_set
    assert all(int(optimizer.state[names[name]]["step"].item()) == 4 for name in added_names)


def test_teacher_factory_must_restore_provider_training_state_and_sampling_counters():
    model = _Model(variant="S-Utility", seed=31)
    provider = _Provider()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("direct_reader.source_read." in name)

    def counter_mutating_trace(provider, model, case_key, key, max_query_rows):
        provider.training_queries += 1
        return _trace_factory(provider, model, case_key, key, max_query_rows)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-5)
    with pytest.raises(ValueError, match="physical sampling counter 'training_queries'"):
        fit_source_read_child_auxiliary(
            model,
            provider,
            optimizer,
            added_parameter_names=_added_names(model),
            train_case_keys=(11, 12, 13),
            max_updates=1,
            parent_sha256="d" * 64,
            source_epoch=500,
            teacher_trace_factory=counter_mutating_trace,
            physical_loss_factory=_physical_loss_factory,
        )

    provider = _Provider()

    def state_mutating_trace(provider, model, case_key, key, max_query_rows):
        provider.calibration_state += 1
        return _trace_factory(provider, model, case_key, key, max_query_rows)

    with pytest.raises(ValueError, match="training_state_dict counters/state"):
        fit_source_read_child_auxiliary(
            model,
            provider,
            optimizer,
            added_parameter_names=_added_names(model),
            train_case_keys=(11, 12, 13),
            max_updates=1,
            parent_sha256="d" * 64,
            source_epoch=500,
            teacher_trace_factory=state_mutating_trace,
            physical_loss_factory=_physical_loss_factory,
        )


def test_auxiliary_fit_rejects_finite_loss_with_nonfinite_new_gradient_before_step(monkeypatch):
    model = _Model(variant="S-Utility", seed=61)
    provider = _Provider()
    names = dict(model.named_parameters())
    added = {name: parameter for name, parameter in names.items()
             if "direct_reader.source_read." in name}
    for name, parameter in names.items():
        parameter.requires_grad_(name in added)

    class GuardedAdamW(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            raise AssertionError("AdamW must not step after a nonfinite source-read gradient.")

    optimizer = GuardedAdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-5)

    def finite_loss_nonfinite_gradient(controller, direct_reader, data, **kwargs):
        del direct_reader, data, kwargs
        parameters = list(controller.parameters())
        loss = torch.sqrt((parameters[0] - parameters[0].detach()).square()).sum()
        for parameter in parameters[1:]:
            loss = loss + parameter.sum() * 0.0
        zero = loss.detach() * 0.0
        return loss, zero, {"scorer": zero, "completion": zero, "total_unweighted": zero}

    monkeypatch.setattr(auxiliary_module, "_scorer_and_completion_losses", finite_loss_nonfinite_gradient)
    with pytest.raises(ValueError, match="nonfinite new-parameter gradients"):
        fit_source_read_child_auxiliary(
            model,
            provider,
            optimizer,
            added_parameter_names=sorted(added),
            train_case_keys=(11, 12, 13),
            max_updates=1,
            parent_sha256="e" * 64,
            source_epoch=500,
            teacher_trace_factory=_trace_factory,
            physical_loss_factory=_physical_loss_factory,
        )
    assert not optimizer.state


def test_initial_auxiliary_fit_uses_fixed_clip_and_error_on_nonfinite(monkeypatch):
    original = torch.nn.utils.clip_grad_norm_
    observations = []

    def observe(parameters, max_norm, *, error_if_nonfinite=False, **kwargs):
        parameters = list(parameters)
        observations.append((len(parameters), max_norm, error_if_nonfinite))
        return original(parameters, max_norm, error_if_nonfinite=error_if_nonfinite, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe)
    _attach("S-Utility", max_updates=3)
    assert observations and all(count > 0 and norm == 1.0 and reject for count, norm, reject in observations)


def test_context_scale_is_rms_dense_context_norm_by_head_on_valid_queries():
    controller = SourceReadControllerV1(
        width=8, coordinate_dim=1, heads=2, thermal=True, completion=False,
    )
    messages = torch.zeros(1, 2, 2, 8)
    messages[0, 0, 0, 0] = 1.0
    messages[0, 0, 0, 4] = 1.0
    messages[0, 0, 1, 1] = 1.0
    messages[0, 1, 0, 2] = 2.0
    messages[0, 1, 0, 6] = 3.0
    alpha = torch.tensor([[[[0.5, 1.0], [0.5, 0.0]], [[1.0, 1.0], [0.0, 0.0]]]])
    case = _TeacherCase(
        case_key="fixture",
        query_states=torch.zeros(1, 2, 8),
        module_states=torch.zeros(1, 2, 8),
        descriptor=torch.zeros(1, 2, 2, 3),
        alpha=alpha,
        fine_messages=messages,
        thermal_k=torch.zeros(1, 2, 2),
        query_mask=torch.tensor([[True, True]]),
        source_present_mask=torch.tensor([[True, True]]),
        local_radius=torch.ones(1, 2),
        distance_to_sources=torch.zeros(1, 2, 2),
        fine_executed_rows=4,
        fine_present_rows=4,
    )
    scales = _context_scales([case], controller)
    # Head 0 norms are sqrt(1/2) and 2; head 1 norms are 1 and 3.
    torch.testing.assert_close(scales, torch.tensor([math.sqrt(2.25), math.sqrt(5.0)]))


def test_train_screen_reports_near_protection_when_nominal_targets_are_infeasible():
    model = _Model(variant="S-Utility", seed=66)
    reader = model.core.direct_reader
    controller = reader.source_read
    cases = []
    for case_key in (11, 12, 13):
        trace = dict(_trace_factory(_Provider(), model, case_key,
                                    SamplingKey(0, 500, case_key, 0, "screen", "S-Utility",
                                                SamplingKey.CASE_EPOCH_VERSION,
                                                "source-read-fixture-train"), 256))
        trace["distance_to_sources"] = torch.zeros_like(trace["distance_to_sources"])
        cases.append(_normalize_teacher_case(case_key, trace, controller, max_query_rows=256))
    scales = _context_scales(cases, controller)
    _, screen = _calibrate_thresholds(
        model, controller, reader, cases, device=torch.device("cpu"), dtype=torch.float32,
        context_scales=scales, heating_envelope=2.5,
    )
    assert screen["target_infeasible_due_to_near_protection"] == [True, True]
    assert screen["actual_union_retention"] == [1.0, 1.0]
    assert screen["actual_physical_pair_retention"] == [1.0, 1.0]
    assert screen["primary_operating_point"] == 0
    assert screen["selection_reason"] == "no_useful_point_default75"


def test_nonthermal_train_screen_uses_null_thermal_metrics_accepted_by_child_validator():
    model = _Model(variant="S-Utility", seed=67)
    reader = model.core.direct_reader
    controller = SourceReadControllerV1(
        width=8, coordinate_dim=1, heads=2, thermal=False, rank=3, scorer_width=6,
    )
    trace = dict(_trace_factory(
        _Provider(), model, 11,
        SamplingKey(0, 500, 0, 0, "nonthermal-screen", "S-Utility",
                    SamplingKey.CASE_EPOCH_VERSION, "source-read-fixture-train"),
        256,
    ))
    trace["thermal_k"] = None
    case = _normalize_teacher_case(11, trace, controller, max_query_rows=256)
    scales = _context_scales([case], controller)

    _, screen = _calibrate_thresholds(
        model, controller, reader, [case], device=torch.device("cpu"), dtype=torch.float32,
        context_scales=scales, heating_envelope=1.0,
    )

    assert screen["thermal_positive_H_endpoint_relative_rms"] is None
    assert screen["thermal_kernel_relative_rms"] is None
    assert screen["thermal_temperature_scale"] is None
    assert screen["thermal_target_scale_convention"] is None
    _validate_train_fidelity_screen(screen)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA calibration regression")
@pytest.mark.parametrize("variant", ["S-Utility", "S-Complete"])
@pytest.mark.parametrize("thermal", [False, True])
def test_cuda_train_screen_accumulates_host_totals_without_optimizer_updates(variant, thermal):
    model = _Model(variant=variant, seed=67)
    reader = model.core.direct_reader
    if not thermal:
        reader.source_read = SourceReadControllerV1(
            width=8, coordinate_dim=1, heads=2, thermal=False, rank=3,
            scorer_width=6, completion=variant == "S-Complete",
        )
    controller = reader.source_read
    trace = dict(_trace_factory(
        _Provider(), model, 11,
        SamplingKey(0, 500, 0, 0, "cuda-screen", variant,
                    SamplingKey.CASE_EPOCH_VERSION, "source-read-fixture-train"),
        256,
    ))
    if not thermal:
        trace["thermal_k"] = None
    case = _normalize_teacher_case(11, trace, controller, max_query_rows=256)
    scales = _context_scales([case], controller)
    cpu_thresholds, cpu_screen = _calibrate_thresholds(
        model, controller, reader, [case], device=torch.device("cpu"), dtype=torch.float32,
        context_scales=scales, heating_envelope=2.5,
    )
    before = copy.deepcopy(model.state_dict())
    model.to("cuda")
    thresholds, screen = _calibrate_thresholds(
        model, controller, reader, [case], device=torch.device("cuda"), dtype=torch.float32,
        context_scales=scales, heating_envelope=2.5,
    )
    _validate_train_fidelity_screen(screen)
    torch.testing.assert_close(torch.tensor(thresholds), torch.tensor(cpu_thresholds),
                               atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(
        torch.tensor(screen["flow_direct_context_relative_rms"]),
        torch.tensor(cpu_screen["flow_direct_context_relative_rms"]),
        atol=3e-6, rtol=3e-5,
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), before[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_teacher_trace_requires_actual_dense_bqm_count_and_separate_present_count():
    model = _Model(variant="S-Utility")
    controller = model.core.direct_reader.source_read
    trace = _trace_factory(_Provider(), model, 12,
                           SamplingKey(0, 500, 0, 0, "fixture", "S-Utility",
                                       SamplingKey.CASE_EPOCH_VERSION, "source-read-fixture-train"),
                           256)
    normalized = _normalize_teacher_case(12, trace, controller, max_query_rows=256)
    assert normalized.fine_executed_rows == 15
    assert normalized.fine_present_rows == 10
    bad = dict(trace, fine_executed_rows=normalized.pair_rows)
    with pytest.raises(ValueError, match="fine_executed_rows"):
        _normalize_teacher_case(12, bad, controller, max_query_rows=256)


def test_physical_calibration_probe_rejects_provider_counter_changes():
    model = _Model(variant="S-Utility")
    provider = _Provider()

    def mutating_factory(provider, model, case_key, key):
        del case_key, key
        provider.training_queries += 1
        return model.core.physical(torch.tensor([[0.4]])).square().mean()

    key = SamplingKey(0, 500, 0, 0, "calibration", "S-Utility",
                      SamplingKey.CASE_EPOCH_VERSION, provider.sampling_dataset_id)
    with pytest.raises(ValueError, match="sampling counter 'training_queries'"):
        _factory_physical_loss(provider, model, 11, key, mutating_factory)
