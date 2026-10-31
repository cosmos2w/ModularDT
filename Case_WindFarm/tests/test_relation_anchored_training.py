from types import SimpleNamespace

import numpy as np
import pytest
import torch

from windfarm.training.content_hypergraph_task import WindContentTargets
from windfarm.training.relation_anchored_task import WindRelationAnchoredTask
from windfarm.workflows.joint_forward import ROLE_NAMES
from test_relation_anchored_wind import _adapter, _case


def test_e64_shapes_have_exact_common_initial_tensors():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = _adapter((4,4,4))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        second = _adapter((8,4,2))
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(torch.equal(first.state_dict()[name], second.state_dict()[name]) for name in first.state_dict())


def test_calibration_replay_accepts_only_exact_panel_and_other_bindings(monkeypatch, tmp_path):
    import json
    from windfarm.training import relation_anchored_task as bridge

    key = {"dataset_fingerprint": "old", "subset_id": "fixed"}
    saved = {"query_digest": "queries", "target_digest": "targets", "scales": [1.0, 2.0]}
    path = tmp_path / "existing.json"
    path.write_text(json.dumps({"cache_key": key}))
    def absent(**kwargs):
        raise FileNotFoundError("strict cache missing")
    monkeypatch.setattr(bridge, "_fit_or_load_role_scales", absent)
    monkeypatch.setattr(bridge, "_read_role_scale_cache", lambda path, key: saved)
    monkeypatch.setattr(bridge, "_role_scale_cache_key", lambda **kwargs: {**key, "dataset_fingerprint": "new"})
    monkeypatch.setattr(bridge, "_fit_train_role_scales", lambda *args, **kwargs: dict(saved))
    args = dict(view=None, manifest={"subset_id": "fixed"}, train_rows=[1], train_layouts=[2],
        training_fingerprint="manifest", normalizer=None, profile=None, catalogue_cache=None,
        cache_dir=tmp_path, allow_cache_write=False)
    calibration, binding = bridge._load_panel_verified_role_scales(existing_cache=path, **args)
    assert calibration == saved and binding["dataset_fingerprint"] == "new"
    assert binding["saved_dataset_fingerprint"] == "old"
    monkeypatch.setattr(bridge, "_fit_train_role_scales", lambda *args, **kwargs: {**saved, "target_digest": "changed"})
    with pytest.raises(ValueError, match="does not reproduce"):
        bridge._load_panel_verified_role_scales(existing_cache=path, **args)
    monkeypatch.setattr(bridge, "_role_scale_cache_key", lambda **kwargs: {**key, "subset_id": "other"})
    with pytest.raises(ValueError, match="recipe or bindings"):
        bridge._load_panel_verified_role_scales(existing_cache=path, **args)


@pytest.mark.parametrize('total_epochs', [200, 1000])
def test_batched_microbatch_loss_backward_and_optimizer_groups(total_epochs):
    model = _adapter().float()
    task = object.__new__(WindRelationAnchoredTask)
    task.model, task.device, task.total_epochs = model, torch.device("cpu"), total_epochs
    task.role_component_scales = {role: np.ones(3) for role in ROLE_NAMES}
    case = _case()
    scene = SimpleNamespace(row_index=case.index, case_name=case.case, layout_index=case.layout_index,
        wind_direction_deg=case.wind_direction_deg, module_centers_D=case.module_centers,
        module_features=case.module_features, module_present=case.module_present,
        module_ids=np.arange(2), operating_features=case.global_context,
        support_lower_D=case.support.lower_D, support_extent_D=case.support.extent_D)
    coordinates = np.array([[[.4,.5,.8], [.6,.7,.8], [.8,.9,.8], [1.,1.,.8], [1.2,1.1,.8]]], dtype=np.float32)
    receivers = SimpleNamespace(coordinates_D=np.concatenate([coordinates, coordinates]),
        flat_indices=np.tile(np.arange(5), (2, 1)), role_ids=np.tile(np.arange(5), (2, 1)))
    target = WindContentTargets(np.zeros((2,5,3), dtype=np.float32), (1,2), ({},{}))
    prediction, auxiliary = task.predict_native(model, (scene, scene), receivers)
    assert any(key.startswith("core_prepare_") for key in auxiliary["work"])
    assert any(key.startswith("core_decode_") for key in auxiliary["work"])
    terms = task.loss_terms(prediction, target, "joint", auxiliary)
    loss = sum(term.weight * term.numerator / term.denominator for term in terms.values())
    loss.backward()
    assert prediction.main_mps.shape == (2,5,3)
    assert torch.isfinite(loss) and all(parameter.grad is not None for parameter in model.parameters())
    groups = task.optimizer_groups(model, "relation_anchored", "joint")
    assert sum(len(group.parameter_names) for group in groups) == len(list(model.named_parameters()))
    assert all(name.endswith('.bias') or 'norm' in name.lower() for name in groups[1].parameter_names)
    assert groups[1].weight_decay == 0
    assert all(group.schedule.total_epochs == total_epochs for group in groups)
    assert all(group.schedule.peak_lr == group.schedule.final_lr == 3e-4 for group in groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.step()
    restored = type(model).from_state_payload(model.state_payload(), state_dict=model.state_dict()).float()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    assert len(restored_optimizer.state) == len(optimizer.state)
    task.model = restored
    replay, _ = task.predict_native(restored, (scene, scene), receivers)
    task.model = model
    updated, _ = task.predict_native(model, (scene, scene), receivers)
    torch.testing.assert_close(replay.main_mps, updated.main_mps)
    task.model = restored
    task.training_samples, task.validation_samples = 7, 3
    state = task.training_state_dict()
    task.training_samples = task.validation_samples = 0
    task.load_training_state_dict(state)
    assert (task.training_samples, task.validation_samples) == (7, 3)


def test_full_train_transform_reuse_requires_all_train_draw_replay(monkeypatch, tmp_path):
    import json
    from windfarm.training import relation_anchored_task as module

    normalizer = SimpleNamespace(to_dict=lambda: {'mean': [1., 2., 3.]})
    profile = SimpleNamespace(to_dict=lambda: {'values': [[4., 5., 6.]]})
    key = {'dataset_profile': 'full', 'subset_manifest_sha256': 'manifest',
           'training_fingerprint': 'manifest', 'train_row_indices': [0, 1, 2],
           'train_layout_indices': [0], 'samples_per_row': 8192, 'profile_bins': 32,
           'seed': 42, 'dataset_fingerprint': 'old', 'train_row_indices_sha256': 'rows'}
    path = tmp_path / 'transforms.json'
    path.write_text(json.dumps({'cache_key': key}))
    def missing(**kwargs):
        raise FileNotFoundError('Current filesystem key is unavailable')
    monkeypatch.setattr(module, '_fit_or_load_formal_transforms', missing)
    monkeypatch.setattr(module, '_read_transform_cache', lambda *args: (normalizer, profile))
    monkeypatch.setattr(module, '_wind_dataset_fingerprint', lambda view: 'new')
    replayed = []
    def replay(volume, rows, **kwargs):
        replayed.append((rows.tolist(), kwargs))
        return normalizer, profile
    monkeypatch.setattr(module, 'fit_velocity_statistics', replay)
    args = dict(existing_cache=path, view=SimpleNamespace(volume=object()),
                train_rows=np.array([0, 1, 2]), train_layouts=[0], manifest={'manifest_sha256': 'manifest'},
                training_fingerprint='manifest', profile_id='full', seed=42,
                cache_dir=tmp_path, allow_cache_write=False)
    _, _, binding = module._load_replayed_full_train_transforms(**args)
    assert replayed == [([0, 1, 2], {'samples_per_row': 8192, 'seed': 42, 'profile_bins': 32})]
    assert binding['dataset_fingerprint'] == 'new' and binding['saved_dataset_fingerprint'] == 'old'
    assert binding['wind_test_target_values_read'] is False
    monkeypatch.setattr(module, 'fit_velocity_statistics', lambda *a, **k: (
        SimpleNamespace(to_dict=lambda: {'mean': [9., 2., 3.]}), profile))
    with pytest.raises(ValueError, match='do not reproduce'):
        module._load_replayed_full_train_transforms(**args)
    with pytest.raises(ValueError, match='full TRAIN recipe'):
        module._load_replayed_full_train_transforms(**{**args, 'train_rows': np.array([0, 1, 9])})


def test_q8192_training_retains_fixed_q4096_validation_and_packing():
    from honf_runtime.unified_training import SamplingKey
    task = object.__new__(WindRelationAnchoredTask)
    task.primary_queries, task.max_microbatch_cases = 8192, 24
    task.validation_query_count, task.validation_microbatch_cases = 4096, 4
    task.validation_rows = np.arange(90)
    task._validation_cache = {}
    task.seed, task.sampling_version, task.sampling_dataset_id = 42, SamplingKey.CASE_EPOCH_VERSION, 'fixed'
    calls = []
    def batch(rows, key, query_count):
        calls.append((list(rows), key.epoch, query_count))
        return tuple(rows)
    task._validation_batch = batch
    panels = list(task.validation_batches())
    assert [len(panel) for panel in panels] == [4] * 22 + [2]
    assert sum(len(rows) for rows, _, _ in calls) == 90
    assert all(epoch == 0 and queries == 4096 for _, epoch, queries in calls)
    assert list(task.validation_batches()) == panels
    assert len(calls) == 23  # Fixed validation samples are reused across reviews.


@pytest.mark.parametrize("shape", [(4,4,4), (8,4,2)])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_ragged_batch_matches_sequential_native_outputs_and_parameter_gradients(shape, batch_size):
    model = _adapter(shape)
    other = type(model).from_state_payload(model.state_payload(), state_dict=model.state_dict()).double()
    first = _case()
    second = _case()
    second.index, second.layout_index = 2, 2
    second.module_centers = np.array([[.15,.25,.8], [.8,1.,.8], [1.5,1.4,.8]])
    second.module_features = np.array([[.5,.8], [.5,.8], [.5,.8]])
    second.module_present = np.array([1.,1.,0.])
    ids = [np.array([5,9]), np.array([3,7,12])]
    coords = torch.tensor([[[.4,.5,.8], [.9,.7,.8]], [[.3,.4,.8], [1.3,1.2,.8]]], dtype=torch.double)
    query_ids = torch.tensor([[19,20],[31,32]])
    cases = [first, second] * (batch_size // 2)
    ids *= batch_size // 2
    coords = coords.repeat(batch_size // 2, 1, 1)
    query_ids = query_ids.repeat(batch_size // 2, 1)
    batch = model.make_scene_batch(cases, module_ids=ids)
    assert batch.module_ids.tolist() == [[5,9,-1],[3,7,12]] * (batch_size // 2)
    assert batch.module_mask.tolist() == [[True,True,False]] * batch_size
    assert torch.equal(batch.environment_measure.sum(-1), torch.full((batch_size,), 8., dtype=torch.double))
    query = model.make_query(batch, coords, query_ids=query_ids)
    assert torch.equal(query.query_ids, query_ids) and bool(query.query_mask.all())
    mask = torch.tensor([[True, False], [False, True]] * (batch_size // 2))
    masked = model.make_query(batch, coords, query_ids=query_ids, query_mask=mask)
    assert torch.equal(masked.query_mask, mask)
    assert bool((masked.anchor_weights[~mask] == 0).all())
    assert bool((masked.anchor_indices[~mask] == -1).all())
    result = model.core.decode(model.core.prepare(batch), query).joint_fields
    sequential = []
    for i, case in enumerate(cases):
        scene = other.make_scene(case, module_ids=ids[i])
        q = other.make_query(scene, coords[i], query_ids=query_ids[i:i+1])
        sequential.append(other.core.decode(other.core.prepare(scene), q).joint_fields)
    expected = torch.cat(sequential)
    torch.testing.assert_close(result, expected, rtol=1e-9, atol=1e-9)
    result.square().sum().backward(); expected.square().sum().backward()
    for (name, parameter), (other_name, reference) in zip(model.named_parameters(), other.named_parameters()):
        assert name == other_name and (parameter.grad is None) == (reference.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-8, atol=1e-9, msg=name)
