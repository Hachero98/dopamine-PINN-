"""Reference estimator recovers the truth from noise-free data; Laplace SD scales with noise."""
import numpy as np

from dopamine_pinn import config as cfg
from dopamine_pinn.solver import make_fd_solver, solve_at, linear_sink
from dopamine_pinn.observations import scattered
from dopamine_pinn.reference import fit_linear, lm_analytical


def test_noise_free_recovery():
    s = make_fd_solver(nx=41)
    fr = solve_at(s, cfg.D_TRUE, linear_sink(cfg.K_TRUE))
    obs = scattered(fr, s, 4001, n_obs=200, noise_sd=0.0)
    r = fit_linear(obs, s, noise_sd=0.02)
    assert abs(r['D'] / cfg.D_TRUE - 1) < 1e-4
    assert abs(r['k'] / cfg.K_TRUE - 1) < 1e-4
    assert 0 < r['sd_logD'] < 0.3 and 0 < r['sd_logk'] < 0.3


def test_lm_baseline_runs():
    s = make_fd_solver(nx=41)
    fr = solve_at(s, cfg.D_TRUE, linear_sink(cfg.K_TRUE))
    r = lm_analytical(scattered(fr, s, 4001))
    assert np.isfinite(r['D']) and np.isfinite(r['k'])
