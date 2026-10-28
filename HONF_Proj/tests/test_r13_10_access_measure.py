"""Access measure equations, discretization, and saved native execution."""
from io import BytesIO

import pytest
import torch

from honf_forward_core.interaction_reused_member_routing.anchored import (
    AnchoredMemberRoutingCore, build_e_access_measure, compile_e_access_directory,
)
from tests.test_e_anchored_packed_execution import fixture


def _directory(m_mask, e_mask, areas, em, ee, coverage=0.95):
    measure = build_e_access_measure(m_mask, e_mask, areas, dtype=areas.dtype, access_measure="mean_one")
    m = m_mask.shape[1]
    return compile_e_access_directory(em, ee, measure[:, :m], measure[:, m:], m_mask, e_mask,
        torch.tensor([[20, 10][:m]], dtype=torch.long), torch.arange(e_mask.shape[1])[None] + 100, retained_mass=coverage)


def test_active_positive_measure_empty_m_padding_and_singleton():
    m = torch.zeros((1, 0), dtype=torch.bool)
    e = torch.tensor([[True, True, True, False]])
    area = torch.tensor([[1., 3., 0., 100.]])
    measure = build_e_access_measure(m, e, area, dtype=area.dtype, access_measure="mean_one")
    torch.testing.assert_close(measure, torch.tensor([[0.5, 1.5, 0., 0.]]))
    directory = _directory(m, e, area, torch.empty(1, 4, 0), torch.zeros(1, 4, 4))
    assert directory.full.row(0, 0).original_id.tolist() == [100]
    assert directory.full.row(0, 0).structural_self.tolist() == [1.]
    assert directory.full.row(0, 2).member_index.numel() == 0
    assert directory.full.row(0, 3).member_index.numel() == 0
    one = build_e_access_measure(m, torch.tensor([[True]]), torch.tensor([[7.]]),
                                  dtype=torch.float32, access_measure="mean_one")
    torch.testing.assert_close(one, torch.ones_like(one))


def test_p_equal_mass_ties_nested_prefixes_and_self():
    m = torch.tensor([[True, True]])
    e = torch.tensor([[True, True, True]])
    area = torch.ones((1, 3))
    em = torch.ones((1, 3, 2))
    ee = torch.ones((1, 3, 3)); ee[:, :, 0] = 0
    rows = [_directory(m, e, area, em, ee, c).restricted.row(0, 0) for c in (0.5, 0.75, 1.)]
    assert rows[-1].original_id.tolist() == [10, 20, 100, 101, 102]
    for low, high in zip(rows, rows[1:]):
        assert set(low.member_index.tolist()) <= set(high.member_index.tolist())
    assert all(100 in row.original_id.tolist() for row in rows)
    assert all(row.access_weight.sum() >= c - 1e-6 for row, c in zip(rows, (0.5, 0.75, 1.)))


def test_refinement_conserves_area_but_changes_p_joint_access():
    # Same self and donor gate; split one identical donor into two children.
    m = torch.tensor([[True]])
    def donor_mass(areas):
        n = len(areas)
        e = torch.ones((1, n), dtype=torch.bool)
        ee = torch.ones((1, n, n)); ee[:, :, 0] = 0
        row = _directory(m, e, torch.tensor([areas]), torch.ones(1, n, 1), ee, 1.).full.row(0, 0)
        return row.access_weight[(~row.is_module) & (row.structural_self == 0)].sum()
    coarse = donor_mass([1., 1.])
    refined = donor_mass([1., .5, .5])
    torch.testing.assert_close(coarse, torch.tensor(1/3))
    torch.testing.assert_close(refined, torch.tensor(3/8))
    assert refined > coarse


@pytest.mark.parametrize("locator", ["e_anchor", "local_e_cover"])
def test_model_choice_saved_reloaded_and_native_read(locator):
    _, scene, query = fixture(locator)
    model = AnchoredMemberRoutingCore(fixture(locator)[0].get_config()["base_model_config"],
          locator_mode=locator, query_tile_size=4, access_measure="mean_one").double().eval()
    data = BytesIO()
    torch.save({"config": model.get_config(), "state": model.state_dict()}, data)
    data.seek(0)
    saved = torch.load(data, weights_only=True)
    restored = AnchoredMemberRoutingCore.from_config(saved["config"]).double().eval()
    restored.load_state_dict(saved["state"])
    assert restored.access_measure == "mean_one"
    before = model.prepare(scene); after = restored.prepare(scene)
    for attr in ("anchored_applied_EM", "anchored_applied_EE"):
        torch.testing.assert_close(getattr(before, attr), getattr(after, attr))
    for mode in ("full", "restricted"):
        expected = model.decode(before, query, intervention={"read_mode": mode}).joint_fields
        actual = restored.decode(after, query, intervention={"read_mode": mode}).joint_fields
        torch.testing.assert_close(actual, expected)
    legacy = model.get_config(); legacy.pop("access_measure")
    assert AnchoredMemberRoutingCore.from_config(legacy).access_measure == "inventory_unit_total"


def test_p_only_reweights_directory_and_keeps_contextual_exchange():
    historical, scene, _ = fixture("e_anchor")
    candidate = AnchoredMemberRoutingCore(historical.get_config()["base_model_config"],
        query_tile_size=4, access_measure="mean_one").double()
    candidate.load_state_dict(historical.state_dict())
    old, new = historical.prepare(scene), candidate.prepare(scene)
    for attr in ("anchored_applied_EM", "anchored_applied_EE", "member_keys_M", "member_keys_E"):
        torch.testing.assert_close(getattr(old, attr), getattr(new, attr), rtol=0, atol=0)
    row = new.e_access_directory.full.row(0, 0)
    assert (row.measure[row.is_module] == 1).all()
    nonzero = row.access_weight > 0
    raw = (row.applied_exchange + row.structural_self) * row.measure
    torch.testing.assert_close(row.access_weight[nonzero], raw[nonzero] / raw.sum())
