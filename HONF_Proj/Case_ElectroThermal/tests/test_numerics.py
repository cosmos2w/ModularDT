import numpy as np
from scipy.sparse.linalg import spsolve

from hetmf.finite_volume import (apply_operator, assemble_matrix, face_energy_heat,
                                 prepare_faces, staggered_face_density)


def test_non_square_face_transmissibility_and_neumann_boundaries():
    c = np.full((3, 5), 2.0)
    tx, ty = prepare_faces(c)
    np.testing.assert_allclose(tx, 2 * 5 / 3)
    np.testing.assert_allclose(ty, 2 * 3 / 5)
    v = np.arange(15.0).reshape(3, 5)
    reaction = np.full_like(v, 0.03)
    sparse = assemble_matrix(tx, ty, reaction)
    np.testing.assert_allclose(sparse @ v.ravel(), apply_operator(v, tx, ty, reaction).ravel())
    flux_x, flux_y = staggered_face_density(v, tx, ty)
    assert not flux_x[:, (0, -1)].any()
    assert not flux_y[(0, -1), :].any()
    np.testing.assert_allclose(apply_operator(np.ones_like(v), tx, ty, np.zeros_like(v)), 0)


def test_harmonic_jump_and_resistance_weighted_face_energy():
    coeff = np.array([[1.0, 4.0], [1.0, 4.0]])
    potential = np.array([[1.0, 0.0], [1.0, 0.0]])
    tx, ty = prepare_faces(coeff)
    np.testing.assert_allclose(tx, 1 / (0.5 / 1 + 0.5 / 4))
    q, power_x, power_y = face_energy_heat(potential, coeff, tx, ty)
    np.testing.assert_allclose(power_y, 0)
    np.testing.assert_allclose(q.sum() / 4, power_x.sum())
    np.testing.assert_allclose(q[:, 0] / q[:, 1], 4)
    assert (q >= 0).all()


def test_smooth_neumann_manufactured_electrical_and_thermal_operators():
    """A Neumann cosine source has second-order cell-centered convergence."""
    errors = []
    for n in (24, 48):
        centers = (np.arange(n) + 0.5) / n
        x, y = np.meshgrid(centers, centers)
        exact = np.cos(np.pi * x) * np.cos(np.pi * y)
        equation_errors = []
        for coefficient, reaction_density in ((1.0, 2.0), (0.7, 5.0)):
            fx, fy = prepare_faces(np.full((n, n), coefficient))
            reaction = np.full((n, n), reaction_density / n**2)
            rhs = (2 * np.pi**2 * coefficient + reaction_density) * exact / n**2
            approximate = spsolve(assemble_matrix(fx, fy, reaction), rhs.ravel()).reshape(n, n)
            equation_errors.append(np.linalg.norm(approximate - exact) / np.linalg.norm(exact))
        errors.append(equation_errors)
    assert all(errors[1][i] < errors[0][i] / 3.5 for i in (0, 1))
