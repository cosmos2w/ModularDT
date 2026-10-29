"""CPU contract checks through the actual general HONF, with no fitting."""

import copy
from dataclasses import fields

import numpy as np
import pytest
import torch

from hetmf.honf_adapter import ElectroThermalHONFAdapter, FIELD_ORDER


def _sample(module_count, *, query_count=13, padded=16):
    t = (np.arange(12) + 0.5) / 12
    x, y = np.meshgrid(t, t)
    environment = np.column_stack((x.ravel(), y.ravel(), np.full((144, 2), 1 / 12),
                                   np.full(144, 1 / 144), np.ones((144, 3))))
    contacts = np.full((padded, 5), np.nan)
    contacts[:module_count] = np.column_stack((np.linspace(0.1, 0.9, module_count),
                                             np.full(module_count, 0.5), np.full(module_count, 0.025),
                                             np.linspace(0.5, 2, module_count), np.linspace(-0.5, 0.5, module_count)))
    mask = np.arange(padded) < module_count
    q = np.column_stack((np.linspace(0.02, 0.98, query_count), np.full(query_count, 0.45)))
    return {"inputs": {"environment_records": environment, "contacts": contacts,
                       "contact_mask": mask, "contact_ids": np.arange(padded)},
            "query_xy": q, "fields": np.zeros((query_count, len(FIELD_ORDER))),
            "query_weight": np.full(query_count, 1 / query_count),
            "metadata": {"state_id": f"fixture-m{module_count}", "split": "train", "seed": 991}}


@pytest.fixture
def adapter():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield ElectroThermalHONFAdapter()
    torch.set_num_threads(old_threads)


def _predict(adapter, model, samples):
    batch = adapter.collate(samples)
    return model.decode(model.prepare(batch.scene), batch.query).nonlinear_field, batch


def test_real_core_mixed_contact_counts_forward_backward_no_parameter_update(adapter):
    model = adapter.build_model(context_dim=16, num_heads=2)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    output, batch = _predict(adapter, model, [_sample(2), _sample(16)])
    assert output.shape == (2, 13, 5)
    assert batch.scene.module_mask.sum(dim=1).tolist() == [2, 16]
    assert torch.isfinite(batch.scene.module_features).all()
    assert batch.scene.module_measure[0, 2:].eq(0).all()
    loss = ((output - batch.targets).square().mean(dim=-1) * batch.query_weights).sum(dim=1).mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    assert model.thermal_response_rank == 0 and model.output_dim == len(FIELD_ORDER)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_complete_contact_permutation_and_padding_are_prediction_invariant(adapter):
    model = adapter.build_model(context_dim=16, num_heads=2).eval()
    original = _sample(16)
    permuted = copy.deepcopy(original)
    permutation = np.random.default_rng(9).permutation(16)
    for key in ("contacts", "contact_mask", "contact_ids"):
        permuted["inputs"][key] = permuted["inputs"][key][permutation]
    with torch.no_grad():
        before, _ = _predict(adapter, model, [original])
        after, _ = _predict(adapter, model, [permuted])
        torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-5)
        small = _sample(2)
        alone, _ = _predict(adapter, model, [small])
        mixed, _ = _predict(adapter, model, [small, original])
        torch.testing.assert_close(alone[0], mixed[0], atol=2e-6, rtol=2e-5)


def test_targets_labels_and_diagnostics_never_enter_scene_or_query(adapter):
    model = adapter.build_model(context_dim=16, num_heads=2).eval()
    sample = _sample(3)
    changed = copy.deepcopy(sample)
    changed["fields"] += 100
    changed["metadata"].update(split="test", seed=-123, terminal_current=99, solver_residual=2)
    changed["inputs"]["secret_target"] = np.ones((13, 5))
    with torch.no_grad():
        before, original_batch = _predict(adapter, model, [sample])
        after, changed_batch = _predict(adapter, model, [changed])
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert not torch.equal(original_batch.targets, changed_batch.targets)
    for attribute in ("module_features", "environment_features", "module_coordinates", "module_measure"):
        torch.testing.assert_close(getattr(original_batch.scene, attribute), getattr(changed_batch.scene, attribute), atol=0, rtol=0)


def test_arbitrary_queries_and_chunked_decode_match(adapter):
    model = adapter.build_model(context_dim=16, num_heads=2).eval()
    batch = adapter.collate([_sample(4, query_count=29)])
    with torch.no_grad():
        prepared = model.prepare(batch.scene)
        full = model.decode(prepared, batch.query).nonlinear_field
        pieces = []
        for start in range(0, 29, 7):
            query = adapter.make_queries(batch.scene, batch.query.coordinates[:, start:start+7].numpy())
            pieces.append(model.decode(prepared, query).nonlinear_field)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=2e-6, rtol=2e-5)


def test_invalid_inputs_are_rejected_before_general_core(adapter):
    bad = _sample(2)
    bad["inputs"]["contacts"][0, 3] = 0
    with pytest.raises(ValueError, match="positive radius and conductance"):
        adapter.collate([bad])
    bad = _sample(2)
    bad["inputs"]["contact_mask"] = bad["inputs"]["contact_mask"].astype(float)
    with pytest.raises(ValueError, match="boolean"):
        adapter.collate([bad])
    bad = _sample(2)
    bad["query_xy"][0, 0] = 1.1
    with pytest.raises(ValueError, match="unit square"):
        adapter.collate([bad])


def test_batch_to_cpu_preserves_every_tensor_contract_and_output(adapter):
    model = adapter.build_model(context_dim=16, num_heads=2).eval()
    sample = _sample(2)
    sample["metadata"]["native_cell_ids"] = np.arange(len(sample["query_xy"]))
    batch = adapter.collate([sample, _sample(16)])
    # Exercise optional query ID movement as well as mandatory scene geometry.
    batch = type(batch)(batch.scene, adapter.make_queries(batch.scene, batch.query.coordinates.numpy(),
                       query_ids=np.tile(np.arange(13), (2, 1))), batch.targets, batch.query_weights, batch.metadata)
    moved = batch.to("cpu")
    assert moved is not batch and moved.scene is not batch.scene and moved.query is not batch.query
    assert moved.metadata is batch.metadata
    for before, after in ((batch.scene, moved.scene), (batch.query, moved.query)):
        assert type(before) is type(after)
        for field in fields(before):
            old_value, new_value = getattr(before, field.name), getattr(after, field.name)
            if torch.is_tensor(old_value):
                assert new_value.device.type == "cpu" and new_value.dtype == old_value.dtype
                torch.testing.assert_close(new_value, old_value, atol=0, rtol=0)
            else:
                assert new_value is old_value
    torch.testing.assert_close(moved.targets, batch.targets, atol=0, rtol=0)
    torch.testing.assert_close(moved.query_weights, batch.query_weights, atol=0, rtol=0)
    with torch.no_grad():
        before = model.decode(model.prepare(batch.scene), batch.query).nonlinear_field
        after = model.decode(model.prepare(moved.scene), moved.query).nonlinear_field
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_literal_source_ids_follow_records_under_permutation_and_subset(adapter):
    sample = _sample(4)
    sample["inputs"].pop("contact_ids")
    sample["metadata"]["contact_ids"] = tuple(["source-a", "source-z", "source-3", "source-4"] + [""] * 12)
    original = adapter.collate([sample])
    changed = copy.deepcopy(sample)
    permutation = np.array([3, 1, 0, 2, *range(4, 16)])
    for key in ("contacts", "contact_mask"):
        changed["inputs"][key] = changed["inputs"][key][permutation]
    changed["metadata"]["contact_ids"] = tuple(np.asarray(changed["metadata"]["contact_ids"])[permutation])
    permuted = adapter.collate([changed])
    torch.testing.assert_close(permuted.scene.module_ids[0], original.scene.module_ids[0, permutation[:4]], rtol=0, atol=0)
    changed["inputs"]["contact_mask"][1:] = False
    subset = adapter.collate([changed])
    assert subset.scene.module_ids[0, 0] == original.scene.module_ids[0, 3]
    bad = copy.deepcopy(sample)
    bad["metadata"]["contact_ids"] = tuple(["repeated"] * 16)
    with pytest.raises(ValueError, match="unique"):
        adapter.collate([bad])
