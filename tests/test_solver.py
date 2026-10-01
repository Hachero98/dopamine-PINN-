"""FD reference solver: agreement with the original NumPy scheme, the analytical solution,
exact mass balance, obstacles, and the Michaelis-Menten limit."""
import numpy as np
import jax.numpy as jnp
import pytest

from dopamine_pinn import config as cfg
from dopamine_pinn.solver import (make_fd_solver, solve_at, linear_sink, mm_sink, interp_frames, total_mass,
                                  analytical_infinite, fd_reference_np, disc_mask, gaussian_ic, grid)


@pytest.fixture(scope='module')
def lin41():
    s = make_fd_solver(nx=41)
    return s, np.asarray(solve_at(s, cfg.D_TRUE, linear_sink(cfg.K_TRUE)))


def test_matches_numpy_scheme(lin41):
    s, fr = lin41
    _, _, ref = fd_reference_np(nx=41)
    peak = ref[-1].max()
    assert np.max(np.abs(fr[-1] - ref[-1])) < 1e-3 * peak   # same scheme, different time step


def test_mass_decays_exactly(lin41):
    s, fr = lin41
    m = np.asarray(total_mass(jnp.asarray(fr), s))
    n_steps = np.arange(len(m)) * s.stride
    expected = m[0] * (1 - cfg.K_TRUE * s.dt) ** n_steps    # explicit Euler on the uptake term
    assert np.max(np.abs(m - expected) / expected) < 1e-10
    assert abs(m[-1] / m[0] - np.exp(-cfg.K_TRUE * cfg.T)) < 1e-4


def test_early_time_analytical():
    s = make_fd_solver(nx=81)
    fr = np.asarray(solve_at(s, cfg.D_TRUE, linear_sink(cfg.K_TRUE)))
    x = grid(81)
    X, Y = np.meshgrid(x, x, indexing='ij')
    # before the pulse feels the walls (by t = 1 ms the walls already add ~2%)
    for t in (0.2, 0.5):
        C = fr[int(round(t / s.dt_store))]
        A = analytical_infinite(X, Y, t)
        assert np.linalg.norm(C - A) / np.linalg.norm(A) < 0.005


def test_interpolation_exact_at_nodes(lin41):
    s, fr = lin41
    x = grid(41)
    i, j, n = 7, 30, 120
    v = interp_frames(jnp.asarray(fr), s.dx, jnp.array([x[i]]), jnp.array([x[j]]), jnp.array([n * s.dt_store]))
    assert abs(float(v[0]) - fr[n, i, j]) < 1e-12


def test_obstacles_conserve_mass_without_uptake():
    discs = [(1.2, 0.0, 0.5), (-1.0, 1.0, 0.4)]
    s = make_fd_solver(nx=61, mask=disc_mask(61, discs), T_end=10.0)
    fr = s(cfg.D_TRUE, linear_sink(0.0), gaussian_ic(61))
    m = np.asarray(total_mass(fr, s))
    assert np.max(np.abs(m / m[0] - 1)) < 1e-12
    assert np.all(np.asarray(fr)[:, s.mask == 0] == 0)


def test_mm_reduces_to_linear_at_large_km(lin41):
    s, fr = lin41
    Km = 1e6
    fr_mm = np.asarray(solve_at(s, cfg.D_TRUE, mm_sink(cfg.K_TRUE * Km, Km)))
    assert np.max(np.abs(fr_mm - fr)) < 1e-6 * fr.max()
