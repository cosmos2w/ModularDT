import copy

import numpy as np

from hetmf.contacts import rasterize_contacts
from hetmf.environment import generate_environment, render_environment, render_points
from hetmf.layouts import generate_layout, validate_contacts
from hetmf.physics import solve_cpu, terminal_conductance_matrix


def _scene():
    env = generate_environment(71, "correlated_mosaic", contrast_s=12, rho=0.7)
    layout = generate_layout(22, 4, "dispersed", "multilevel")
    return {"state_id": "unit", "environment": env, "contacts": layout["contacts"]}


def test_log_bilinear_exact_controls_border_and_bounds():
    env = generate_environment(1, "inclusions")
    x = (np.arange(12) + 0.5) / 12
    xx, yy = np.meshgrid(x, x)
    rendered = render_points(env, xx, yy)
    for key, values in zip(("s", "kappa", "beta"), rendered):
        np.testing.assert_allclose(values, env[key], rtol=2e-15)
    corners = render_points(env, np.array([0, 1]), np.array([0, 1]))
    for key, values in zip(("s", "kappa", "beta"), corners):
        catalogue = np.asarray(env[key])
        np.testing.assert_allclose(values, [catalogue[0, 0], catalogue[-1, -1]])
    for n in (48, 96, 192):
        for key, values in zip(("s", "kappa", "beta"), render_environment(env, n, n)):
            control = np.asarray(env[key])
            assert values.min() >= control.min() - 1e-14
            assert values.max() <= control.max() + 1e-14


def test_contacts_normalize_and_layouts_are_legal():
    for m in (2, 3, 4, 6, 8, 12, 16):
        for family in ("dispersed", "rows", "two_cluster"):
            layout = generate_layout(17, m, family, "multilevel")
            assert validate_contacts(layout["contacts"]) >= 0.015 - 1e-12
    contacts = generate_layout(13, 4, "dispersed")["contacts"]
    for n in (48, 96):
        for order in (2, 4, 8):
            chi, integrals = rasterize_contacts(contacts, n, n, order)
            np.testing.assert_allclose(chi.sum(axis=(1, 2)) / n**2, 1, atol=1e-14)
            assert np.max(np.abs(integrals - 1)) < 2e-3


def test_cpu_balances_and_exact_voltage_laws():
    scene = _scene()
    base = solve_cpu(scene, 32)
    d = base["diagnostics"]
    for name in ("terminal_current_balance", "electrical_energy_balance", "thermal_energy_balance"):
        assert abs(d[name]["relative"]) < 1e-11
    assert d["electrical_residual"]["relative_l2"] < 1e-11
    assert d["thermal_residual"]["relative_l2"] < 1e-11
    assert (base["q_sheet"] >= 0).all() and (base["q_contact"] >= 0).all()
    shifted = copy.deepcopy(scene)
    for contact in shifted["contacts"]:
        contact["u"] += 0.37
    shifted_out = solve_cpu(shifted, 32)
    np.testing.assert_allclose(shifted_out["v"], base["v"] + 0.37, atol=1e-12)
    for key in ("theta", "q", "j_face_x", "j_face_y", "terminal_currents"):
        np.testing.assert_allclose(shifted_out[key], base[key], atol=1e-12)
    scaled = copy.deepcopy(scene)
    for contact in scaled["contacts"]:
        contact["u"] *= -1.8
    scaled_out = solve_cpu(scaled, 32)
    for key in ("v", "j_face_x", "j_face_y", "terminal_currents"):
        np.testing.assert_allclose(scaled_out[key], -1.8 * base[key], atol=1e-12)
    for key in ("theta", "q", "q_sheet", "q_contact"):
        np.testing.assert_allclose(scaled_out[key], 1.8**2 * base[key], atol=1e-12)


def test_thermal_null_permutation_and_equal_voltage():
    scene = _scene()
    base = solve_cpu(scene, 24)
    changed = copy.deepcopy(scene)
    changed["environment"]["kappa"] = (2 * np.asarray(changed["environment"]["kappa"])).tolist()
    changed["environment"]["beta"] = (3 * np.asarray(changed["environment"]["beta"])).tolist()
    other = solve_cpu(changed, 24)
    for key in ("v", "terminal_currents", "q"):
        np.testing.assert_allclose(other[key], base[key], atol=1e-12)
    assert np.max(np.abs(other["theta"] - base["theta"])) > 1e-5
    permuted = copy.deepcopy(scene)
    permuted["contacts"] = permuted["contacts"][::-1]
    reverse = solve_cpu(permuted, 24)
    np.testing.assert_allclose(reverse["v"], base["v"], atol=1e-12)
    np.testing.assert_allclose(reverse["theta"], base["theta"], atol=1e-12)
    np.testing.assert_allclose(reverse["terminal_currents"], base["terminal_currents"][::-1], atol=1e-12)
    equal = copy.deepcopy(scene)
    for contact in equal["contacts"]:
        contact["u"] = 0.41
    zero = solve_cpu(equal, 24)
    np.testing.assert_allclose(zero["v"], 0.41, atol=1e-12)
    assert np.max(np.abs(zero["q"])) < 1e-24
    assert np.max(np.abs(zero["theta"])) < 1e-24
    assert zero["diagnostics"]["electrical_energy_balance"]["relative"] is None


def test_terminal_basis_matrix_is_reciprocal_and_semidefinite():
    basis = terminal_conductance_matrix(_scene(), 24)
    assert basis["basis_solves"] == 4
    assert basis["max_relative_residual"] < 1e-11
    assert basis["max_symmetry_error"] < 1e-11
    assert basis["max_row_sum_error"] < 1e-11
    assert basis["min_symmetric_eigenvalue"] > -1e-11
