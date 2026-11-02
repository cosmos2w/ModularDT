from types import SimpleNamespace as NS

import numpy as np
import pytest

from channelthermal.training import thermal_response_packs as packs


def _store():
    members = tuple(NS(module_id=f"0001:module:{i}", position_xy=(2. + i, 3.), heating=1.) for i in range(2))
    context = NS(values={"nu": .01, "solid_alpha": .01, "fluid_alpha": .02, "solid_k": 1.,
                        "fluid_k": 1., "module_radius": .45, "re": 50., "u_in": 1.,
                        "domain_length_x": 12., "domain_length_y": 6.})
    def record(shift):
        def role(values, features, channels, ids=None):
            return NS(values=values, query_features=features, channel_names=channels,
                      query_ids=tuple(map(str, range(len(values)))), receiver_module_ids=ids,
                      valid_mask=np.ones(values.shape, dtype=bool))
        roles = {
            "fluid_fields": role(np.zeros((600, 5)) + shift, np.zeros((600, 2)), ("u", "v", "p", "omega", "temperature")),
            "solid_temperature": role(np.zeros((128, 1)) + shift, np.tile(np.arange(128).reshape(64, 2), (2, 1)), ("temperature",),
                                      tuple(m.module_id for m in members for _ in range(64))),
            "interface": role(np.zeros((128, 2)) + shift, np.tile(np.arange(192).reshape(64, 3), (2, 1)), ("T_surface", "q_normal"),
                              tuple(m.module_id for m in members for _ in range(64))),
        }
        return NS(design=NS(active_modules=members, active_module_ids=tuple(m.module_id for m in members)),
                  context=context, output=NS(roles=roles))
    base = record(0.)
    variants = {label: record(i + 1.) for i, label in enumerate((*packs.SINGLE_EDITS, *packs.HEAT_EDITS))}
    variants["i_minus"].output.roles["interface"].valid_mask[0, 1] = False
    stencil = NS(baseline=base, variants=variants)
    return packs.FitResponsePacks({"0001": stencil}, {}, np.ones(4))


def test_response_pack_keeps_q_mask_and_shared_material_coordinates():
    store = _store()
    sample = packs.sample_fit_pack(store, "0001", 1)
    valid = sample["deltas"]["geometry"]["interface"]["valid_mask"]
    assert valid[0, 0] and not valid[0, 1]
    assert sample["material_local"].shape == (32, 2)
    points = sample["roles"]["baseline"]["solid_temperature"]["query_features"]
    np.testing.assert_array_equal(points[:32], points[32:])
    assert len(sample["fluid_indices"]) == 512
    assert set(sample["structures"]["geometry"]) >= {"heat_powers", "module_source_ids", "module_centers"}


def test_response_draw_does_not_consume_global_rng_or_depend_on_it():
    store = _store()
    np.random.seed(27)
    before = np.random.get_state()
    first = packs.sample_fit_pack(store, "0001", 2)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    np.random.seed(31)
    second = packs.sample_fit_pack(store, "0001", 2)
    np.testing.assert_array_equal(first["fluid_indices"], second["fluid_indices"])
    np.testing.assert_array_equal(first["material_indices"], second["material_indices"])
    with pytest.raises(ValueError, match="FIT"):
        packs.sample_fit_pack(store, "0304", 2)
