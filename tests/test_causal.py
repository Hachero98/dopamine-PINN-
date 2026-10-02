"""Causal layer: design, exact effect structure, identifiability, staged training."""
import numpy as np
import jax.numpy as jnp

from dopamine_pinn import config as cfg, benchmark
from dopamine_pinn import causal as ca
from dopamine_pinn.solver import make_fd_solver
from dopamine_pinn.pinn import TrainConfig, sample_points


def test_untreated_design_is_the_benchmark_set():
    s = make_fd_solver()
    obs = ca.untreated(ca.dose_observations(s))
    ref, _ = benchmark.load('linear_s4001')
    for k in ('x_d', 'y_d', 't_d', 'C_d'):
        assert np.allclose(obs[k], ref[k], rtol=0, atol=1e-12)


def test_heldout_dose_never_observed():
    obs = ca.dose_observations(make_fd_solver(nx=41), n_obs=20)
    assert ca.A_HELDOUT not in set(obs['a_d'].tolist())
    assert sorted(set(obs['a_d'].tolist())) == list(ca.A_TRAIN)


def test_effect_matches_closed_form():
    """Linear uptake: tau_a = C^(0) (exp(k eta a t) - 1) up to the time-step error."""
    s = make_fd_solver(nx=41)
    a = ca.A_HELDOUT
    f0, fa = np.asarray(ca.truth_frames(0.0, s)), np.asarray(ca.truth_frames(a, s))
    t = np.arange(f0.shape[0]) * s.dt_store
    tau_cf = f0 * (np.exp(cfg.K_TRUE * ca.ETA_TRUE * a * t) - 1.0)[:, None, None]
    assert np.linalg.norm((fa - f0) - tau_cf) / np.linalg.norm(tau_cf) < 1e-3


def test_eta_unidentifiable_from_untreated_data():
    s = make_fd_solver(nx=41)
    idf = ca.identifiability(ca.dose_observations(s, n_obs=60), s)
    assert idf['untreated_only']['unidentifiable'] == ['eta']
    assert idf['all_training_doses']['unidentifiable'] == []


def test_staged_training_pins_parameters():
    s = make_fd_solver(nx=41)
    obs = ca.dose_observations(s, n_obs=30)
    tc = TrainConfig.quick()
    net = ca.DoseNet(layers=(4, 16, 16, 1))
    pts = sample_points(n_domain=200, n_initial=40, n_boundary=40)
    hist, p, _ = ca.run_stages(obs, tc, net=net, pts=pts)
    assert hist['stage2']['params']['eta'] == ca.ETA_INIT
    for q in ('D', 'k'):
        assert hist['stage3']['params'][q] == hist['stage2']['params'][q]
    assert set(p) == {'net', 'logD', 'logk', 'logeta'}
    C = ca.predict_dose(net, p, np.zeros(2), np.zeros(2), np.array([1.0, 2.0]), ca.A_HELDOUT)
    assert np.all(np.isfinite(C))
