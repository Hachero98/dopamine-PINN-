"""One complete study: data, reference, bound, staged PINN, held-out counterfactual.

    res = run_study('linear', 'fast', obs_seed=4001, out='runs')

Writes runs/<mechanism>/<profile>/obs<seed>_net<seed>/
    results.json  every number (parameters, errors, bounds, effects, diagnostics)
    weights.pkl   network weights and fitted log-parameters, so fields and figures can be
                  rebuilt without retraining (load_study)
A finished study is loaded instead of rerun.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from . import mechanisms, reference
from .settings import Design, PROFILES
from .model import Forward, observations
from .effects import EffectGrid, scores, counterfactual_bound
from .pinn import staged, DoseNet
from dopamine_pinn import config as cfg

DOSE_GRID = np.linspace(0.0, 1.0, 11)


def study_dir(out, mech_name, profile_name, obs_seed, net_seed):
    return Path(out) / mech_name / profile_name / f'obs{obs_seed}_net{net_seed}'


def _err_pct(mech, values):
    return {p: 100.0 * (values[p] - mech.truth[p]) / mech.truth[p] for p in mech.params}


def _err_log_pct(mech, values):
    return {p: 100.0 * float(np.log(values[p] / mech.truth[p])) for p in mech.params}


def evaluate(fwd, grid, net, p_net, theta_pinn, ref4, a):
    """Held-out effect three ways (network; PDE at PINN parameters; PDE at reference
    parameters with delta-method SDs) against the truth, plus the dose-response of GTE."""
    ts, x = grid.ts, grid.x
    Tg, X, Y = np.meshgrid(ts, x, x, indexing='ij')
    fd = lambda th, d: np.asarray(grid.field(th, d))
    th_true = fwd.theta_true
    th_ref = fwd.theta(ref4['params'])
    cov_ref = np.asarray(ref4['cov_log'])

    truth0 = fd(th_true, 0.0)
    true = grid.summaries(truth0, fd(th_true, a))
    net0 = net.predict(p_net, X, Y, Tg, 0.0)
    est = {'network': grid.summaries(net0, net.predict(p_net, X, Y, Tg, a)),
           'pinn_pde': grid.summaries(fd(theta_pinn, 0.0), fd(theta_pinn, a)),
           'reference': grid.summaries(fd(th_ref, 0.0), fd(th_ref, a))}
    g_ref = grid.gradients(th_ref, a)
    sd_gte = float(reference.delta_sd(cov_ref, g_ref['dGTE']))
    sd_ate = reference.delta_sd(cov_ref, g_ref['dATE'])
    metrics = {'truth': {k: true[k] for k in ('GTE', 'peak_tau', 'peak_ATE', 't_peak_ATE')},
               **{k: scores(v, true) for k, v in est.items()}}
    metrics['reference']['GTE_sd'] = sd_gte
    metrics['reference']['GTE_sd_pct'] = 100 * sd_gte / true['GTE']
    metrics['reference']['GTE_covered'] = bool(abs(est['reference']['GTE'] - true['GTE']) <= 1.96 * sd_gte)
    curves = {'ts': ts.tolist(), 'ATE_truth': true['ATE'].tolist(),
              **{f'ATE_{k}': v['ATE'].tolist() for k, v in est.items()}, 'ATE_reference_sd': sd_ate.tolist(),
              'CE_centre_truth': true['CE'][:, len(x) // 2].tolist(),
              **{f'CE_centre_{k}': v['CE'][:, len(x) // 2].tolist() for k, v in est.items()}, 'x': x.tolist()}

    gte = lambda f0, fa: grid.summaries(f0, fa)['GTE']
    dr = {'doses': DOSE_GRID.tolist(), 'truth': [], 'network': [], 'pinn_pde': [], 'reference': [], 'reference_sd': []}
    pinn0, ref0 = fd(theta_pinn, 0.0), fd(th_ref, 0.0)
    for d in DOSE_GRID:
        dr['truth'].append(gte(truth0, fd(th_true, d)))
        dr['network'].append(gte(net0, net.predict(p_net, X, Y, Tg, d)))
        dr['pinn_pde'].append(gte(pinn0, fd(theta_pinn, d)))
        dr['reference'].append(gte(ref0, fd(th_ref, d)))
        g = np.asarray(jax.jacfwd(grid.gte)(jnp.asarray(th_ref), d))
        dr['reference_sd'].append(float(reference.delta_sd(cov_ref, g)))
    return metrics, curves, dr


def diagnostics(fwd, obs, mech, hist, ref4, profile, design):
    """Is the PINN fitting noise (data misfit below the reference's) and how loose is the
    physics (residual RMS against the RMS of the true uptake term over space, time and
    doses 0, 0.5, 1)?"""
    s2 = design.noise_sd**2
    ref_pred = np.asarray(fwd.predictor(obs)(fwd.theta(ref4['params'])))
    q = fwd.q(fwd.theta_true)
    up = np.concatenate([np.ravel(mech.uptake(q, fwd.frames(fwd.theta_true, d), d)) for d in (0.0, 0.5, 1.0)])
    L4 = hist['stage4']['L']
    return {'data_misfit_pinn_over_sigma2': L4['d'] / s2,
            'data_misfit_ref_over_sigma2': float(np.mean((ref_pred - obs['C_d']) ** 2)) / s2,
            'residual_rms': float(np.sqrt(L4['r'])),
            'uptake_rms_truth': float(np.sqrt(np.mean(up ** 2))),
            'residual_over_uptake': float(np.sqrt(L4['r']) / np.sqrt(np.mean(up ** 2))),
            'weighted_data_over_residual': profile.w_data * L4['d'] / max(profile.w_res * L4['r'], 1e-300)}


def run_study(mech_name, profile='fast', obs_seed=4001, net_seed=cfg.SEED, design=Design(), out='runs',
              mech_kw=None, verbose=True, force=False):
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    mech = mechanisms.get(mech_name, **(mech_kw or {}))
    d = study_dir(out, mech_name, prof.name, obs_seed, net_seed)
    if (d / 'results.json').exists() and not force:
        return json.loads((d / 'results.json').read_text())
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    say(f'== {mech_name} | {prof.name} | obs {obs_seed} | net {net_seed}')

    fwd = Forward(mech, design, nx=prof.fd_nx)
    obs = observations(fwd, obs_seed)
    grid = EffectGrid(fwd, prof.eval_dt)
    a = design.heldout

    ref = reference.staged(fwd, obs)
    ref4 = ref['stage4']
    say('  reference:', {p: f'{e:+.2f}%' for p, e in _err_pct(mech, ref4['params']).items()})
    g_true = grid.gradients(fwd.theta_true, a)
    bound = counterfactual_bound(reference.fisher(fwd, obs), g_true, mech.params)
    n_tau = float(np.linalg.norm(g_true['tau']))

    hist, p, net = staged(mech, prof, design, obs, net_seed, verbose=verbose)
    pinn_vals = hist['stage4']['params']
    say('  PINN:     ', {q: f'{e:+.2f}%' for q, e in _err_pct(mech, pinn_vals).items()})
    metrics, curves, dr = evaluate(fwd, grid, net, p['net'], fwd.theta(pinn_vals), ref4, a)
    curves['ATE_bound_sd'] = bound['sd_ATE'].tolist()

    res = {
        'mechanism': mech.to_dict(), 'profile': prof.to_dict(), 'design': design.to_dict(),
        'seeds': {'obs': obs_seed, 'net': net_seed},
        'reference': {s: {'params': r['params'], 'err_pct': _err_pct(mech, r['params']),
                          'err_log_pct': _err_log_pct(mech, r['params']), 'sd_log': r['sd_log'], 'corr': r['corr'],
                          'chi2_p': r['chi2_p'], 'at_bound': r['at_bound']} for s, r in ref.items()},
        'pinn': {s: {**h, 'err_pct': _err_pct(mech, h['params']), 'err_log_pct': _err_log_pct(mech, h['params'])}
                 for s, h in hist.items()},
        'bound': {'fisher_condition': bound['fisher_condition'],
                  'sd_log_pct': {k: 100 * v for k, v in bound['sd_log_params'].items()},
                  'GTE_sd_pct': 100 * bound['sd_GTE'] / float(g_true['GTE']),
                  'tau_L2_rmse_pct': 100 * bound['rmse_L2_tau'] / n_tau},
        'heldout': metrics, 'curves': curves, 'dose_response': dr,
        'diagnostics': diagnostics(fwd, obs, mech, hist, ref4, prof, design),
        'minutes': (time.time() - t0) / 60,
    }
    for k in ('network', 'pinn_pde', 'reference'):
        m = metrics[k]
        say(f'  held-out {k:<9} tau L2 {m["tau_L2_pct"]:6.2f}%  GTE {m["GTE_err_pct"]:+6.2f}%')
    say(f'  bound: GTE {res["bound"]["GTE_sd_pct"]:.2f}%  tau L2 {res["bound"]["tau_L2_rmse_pct"]:.2f}%  '
        f'({res["minutes"]:.1f} min)')

    d.mkdir(parents=True, exist_ok=True)
    (d / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    with open(d / 'weights.pkl', 'wb') as f:
        pickle.dump({'net': jax.device_get(nnx.to_pure_dict(p['net'])),
                     'log': {k: float(v) for k, v in p.items() if k.startswith('log')},
                     'layers': prof.layers}, f)
    return res


def load_study(path):
    """(results, net, p_net) of a saved study; p_net is ready for net.predict."""
    path = Path(path)
    res = json.loads((path / 'results.json').read_text())
    with open(path / 'weights.pkl', 'rb') as f:
        w = pickle.load(f)
    net = DoseNet(tuple(w['layers']))
    state = net.init(0)
    nnx.replace_by_pure_dict(state, w['net'])
    return res, net, state
