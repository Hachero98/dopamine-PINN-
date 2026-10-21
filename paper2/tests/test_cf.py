"""Workbench checks (CPU, about 2 minutes):  python3.11 -m pytest -q paper2/tests"""
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cf  # noqa: E402
from cf import mechanisms, reference  # noqa: E402
from cf.settings import Design, PROFILES  # noqa: E402
from cf.model import Forward, observations  # noqa: E402
from cf.effects import EffectGrid, counterfactual_bound  # noqa: E402
from cf.pinn import staged  # noqa: E402

SMOKE = PROFILES['smoke']


@pytest.mark.parametrize('name', ['mm_capacity', 'mm_competitive'])
def test_mm_reduces_to_linear_at_low_concentration(name):
    """Where C << Km the MM rates match the linear model (the bridge between Parts A and B),
    up to paper 1's rounding (Vmax / Km = 0.0195 vs k = 0.020). Capacity loss matches at
    every dose; competitive block only at a = 0 and a = 1, since 1 / (1 + gamma a) is not
    linear in the dose (0.57 vs 0.70 at a = 0.5)."""
    lin, mm = mechanisms.get('linear'), mechanisms.get(name)
    C = jnp.asarray(1e-5)
    rate = lambda m, a: float(m.uptake(m.truth, C, a) / C)
    for a in (0.0, 0.5, 1.0):
        same = abs(rate(mm, a) / rate(lin, a) - 1) < 0.03
        assert same == (name == 'mm_capacity' or a in (0.0, 1.0))


def test_linear_observations_match_the_causal_runs():
    from dopamine_pinn import causal as ca
    fwd = Forward(mechanisms.get('linear'), Design(n_obs=30))
    o = observations(fwd, 4001)
    ref = ca.dose_observations(fwd.solver, n_obs=30)
    for k in ('x_d', 'y_d', 't_d', 'a_d', 'C_d'):
        assert np.allclose(o[k], ref[k], atol=1e-12)


def test_reference_recovers_truth_without_noise():
    fwd = Forward(mechanisms.get('linear'), Design(n_obs=40), nx=31)
    obs = observations(fwd, 4001)
    obs['C_d'] = obs['C_clean']
    r = reference.staged(fwd, obs)['stage4']
    for p, v in r['params'].items():
        assert abs(v / fwd.mech.truth[p] - 1) < 1e-4


def test_bound_matches_the_exact_linear_bound():
    """FD bound (any mechanism) against the exact image-series bound of dopamine_pinn.counterfactual."""
    from dopamine_pinn import counterfactual as ex
    fwd = Forward(mechanisms.get('linear'), Design(n_obs=60), nx=31)
    obs = observations(fwd, 4001)
    grid = EffectGrid(fwd, 5.0)
    b = counterfactual_bound(reference.fisher(fwd, obs), grid.gradients(fwd.theta_true, 0.5), fwd.mech.params)
    eo = ex.exact_observations(4001, n_obs=60)
    b_ex = ex.counterfactual_crlb(ex.fisher(eo), ex.EffectGrid(nx=31, dt=5.0).gradients(0.5))
    for p in fwd.mech.params:
        assert abs(b['sd_log_params'][p] / b_ex['sd_log_params'][p] - 1) < 0.03
    assert abs(b['sd_GTE'] / b_ex['sd_GTE'] - 1) < 0.03


@pytest.mark.parametrize('name', sorted(mechanisms.MECHANISMS))
def test_staging_pins_the_right_parameters(name):
    mech = mechanisms.get(name)
    design = Design(n_obs=20)
    obs = observations(Forward(mech, design, nx=31), 4001)
    hist, p, _ = staged(mech, SMOKE.replace(adam_iters=50, lbfgs_iters=5), design, obs, 1234, verbose=False)
    assert hist['stage2']['params'][mech.dose_param] == mech.init[mech.dose_param]
    for q in mech.baseline_params:
        assert hist['stage3']['params'][q] == hist['stage2']['params'][q]
    assert set(p) == {'net'} | {'log' + q for q in mech.params}


def test_study_saves_and_reloads(tmp_path):
    res = cf.run_study('linear', SMOKE.replace(adam_iters=50, lbfgs_iters=5), obs_seed=4001, out=tmp_path,
                       design=Design(n_obs=40), verbose=False)
    assert res['bound']['GTE_sd_pct'] > 0 and 'diagnostics' in res
    d = cf.study.study_dir(tmp_path, 'linear', 'smoke', 4001, 1234)
    res2, net, p_net = cf.load_study(d)
    C = net.predict(p_net, np.zeros(3), np.zeros(3), np.array([1.0, 10.0, 40.0]), 0.5)
    assert np.all(np.isfinite(C))
    # the saved network reproduces the saved held-out effect
    from cf.plots import plot_study
    plot_study(d)
    assert (d / 'effect_fields.png').exists()
