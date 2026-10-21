"""Exact linear dose model, counterfactual Cramer-Rao bound and reference fit."""
import numpy as np
import jax.numpy as jnp

from dopamine_pinn import causal as ca, counterfactual as cf
from dopamine_pinn.solver import make_fd_solver


def test_exact_design_matches_fd_design():
    """Same points and noise as causal.dose_observations; values agree to FD accuracy."""
    s = make_fd_solver()
    fd, ex = ca.dose_observations(s, n_obs=50), cf.exact_observations(n_obs=50)
    for k in ('x_d', 'y_d', 't_d', 'a_d'):
        assert np.array_equal(fd[k], ex[k])
    assert np.max(np.abs(fd['C_clean'] - ex['C_clean'])) < 2e-4
    assert np.allclose(fd['C_d'] - fd['C_clean'], ex['C_d'] - ex['C_clean'], atol=1e-12)


def test_fit_recovers_truth_without_noise():
    obs = cf.exact_observations(n_obs=60)
    obs['C_d'] = obs['C_clean']
    r = cf.fit_exact(obs)
    assert np.allclose(r['theta'], cf.TH_TRUE, atol=1e-6)


def test_gte_closed_form_and_no_D_dependence():
    g = cf.EffectGrid(nx=41).gradients(ca.A_HELDOUT)
    assert abs(g['GTE'] / cf.gte_closed_form(ca.A_HELDOUT, grid=cf.EffectGrid(nx=41)) - 1) < 1e-3
    assert abs(g['dGTE'][0]) < 1e-9 * abs(g['dGTE'][1])
    assert abs(g['dtau'][..., 0]).max() > 1e-3 * abs(g['dtau'][..., 1]).max()   # the field does need D


def test_crlb_of_a_parameter_is_the_parameter_bound():
    """g = log k has gradient e_k, so its counterfactual bound is the parameter bound."""
    obs = cf.exact_observations(n_obs=60)
    F = cf.fisher(obs)
    grads = cf.EffectGrid(nx=21).gradients(ca.A_HELDOUT)
    grads['dGTE'] = np.array([0.0, 1.0, 0.0])
    b = cf.counterfactual_crlb(F, grads)
    assert np.isclose(b['sd_GTE'], b['sd_log_params']['k'])


def test_crlb_matches_fd_identifiability():
    s = make_fd_solver(nx=41)
    fd = ca.identifiability(ca.dose_observations(s, n_obs=60), s)['all_training_doses']['crlb_sd_log']
    ex = cf.counterfactual_crlb(cf.fisher(cf.exact_observations(n_obs=60)),
                                cf.EffectGrid(nx=21).gradients(0.5))['sd_log_params']
    for p in ca.PARAMS:
        assert abs(ex[p] / fd[p] - 1) < 0.02


def test_clopper_pearson():
    lo, hi = cf.clopper_pearson(48, 50)
    assert 0.85 < lo < 0.87 and 0.99 < hi < 1.0
    assert cf.clopper_pearson(50, 50)[1] == 1.0
