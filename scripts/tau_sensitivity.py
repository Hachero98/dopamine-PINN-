"""Sensitivity of the measurement-aware reference fit to a misspecified response lag.

The data are the sensor experiment of the main text (two-site design, seeds
5001-5050), generated with the true lag tau = 2 ms and the fine quadrature.
Each record is refitted with the measurement-aware model, but with the lag
assumed to be tau_fit in {1.6, 1.8, 2.0, 2.2, 2.4} ms (-20% to +20%). Everything
else (tip radius, noise correlation time, offset projection) is as in the
experiment. tau_fit = 2.0 reproduces the reference fits of Table 3.

Reports, per scenario and tau_fit: median signed error and RMSE of D, k and
lambda = sqrt(D/k), and coverage of the 95% Laplace intervals (lambda by the
delta method on log lambda = (log D - log k)/2, using the full covariance).

NumPy/SciPy only, a few minutes on a laptop.
Usage:  python scripts/tau_sensitivity.py [measurement_model.json] [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from review_analyses import (D_TRUE, K_TRUE, C0, D_INIT, K_INIT, SIG_N, BOUNDS,
                             MEAS_A, MEAS_TAU, MEAS_TAU_N, MEAS_DT, MEAS_TMIN,
                             MEAS_TMAX, MEAS_PRE, MEAS_SITES, SCEN, exact_C,
                             disk_rule, lag_rule, build_operator, apply_op,
                             block_matrix)

FIT_DTQ = 0.10                     # fit quadrature step of the lag kernel (ms)
DRIFT_SD = 0.02 * C0
SEEDS = list(range(5001, 5051))
TAU_FITS = [1.6, 1.8, 2.0, 2.2, 2.4]
SCENARIOS = ['M2_lag', 'M5_all']
LAM_TRUE = np.sqrt(D_TRUE / K_TRUE)

TIMES = np.r_[np.arange(-MEAS_PRE, 0.0, MEAS_DT), np.arange(MEAS_TMIN, MEAS_TMAX + 1e-9, MEAS_DT)]
N_T = TIMES.size


def disk_rule_fit(a):              # seven-point rule used in fitting
    if a <= 0:
        return np.zeros((1, 2)), np.ones(1)
    th = 2 * np.pi * np.arange(6) / 6 + np.pi / 6
    off = np.vstack([[0.0, 0.0], a * np.sqrt(2 / 3) * np.c_[np.cos(th), np.sin(th)]])
    return off, np.r_[0.25, np.full(6, 0.125)]


def lag_rule_fit(tau):             # trapezoid, 0.1-ms steps over 6 tau
    if tau <= 0:
        return np.zeros(1), np.ones(1)
    s = np.arange(0.0, 6 * tau + 1e-9, FIT_DTQ)
    w = np.exp(-s / tau) / tau * FIT_DTQ
    w[0] *= 0.5; w[-1] *= 0.5
    return s, w * (1 - np.exp(-s[-1] / tau)) / w.sum()


def build_fit_operator(a, tau):
    off, wd = disk_rule_fit(a)
    s, wl = lag_rule_fit(tau)
    X, Y, Tn, W = [], [], [], []
    for sx, sy in MEAS_SITES:
        for tt in TIMES:
            tq = tt - s
            X.append(np.repeat(sx + off[:, 0], s.size))
            Y.append(np.repeat(sy + off[:, 1], s.size))
            Tn.append(np.tile(np.maximum(tq, 0.0), off.shape[0]))
            W.append(np.outer(wd, wl * (tq >= 0)).ravel())
    return np.array(X), np.array(Y), np.array(Tn), np.array(W)


def make_noise(rho_tau, drift, seed):
    """As in dopamine_PINN_tuning.ipynb (same draws per seed)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in MEAS_SITES:
        xi = rng.normal(0.0, 1.0, N_T)
        if rho_tau > 0:
            rho = np.exp(-np.diff(TIMES) / rho_tau)
            e = np.empty(N_T); e[0] = xi[0]
            for i in range(1, N_T):
                e[i] = rho[i - 1] * e[i - 1] + np.sqrt(1 - rho[i - 1] ** 2) * xi[i]
        else:
            e = xi
        e = SIG_N * e
        if drift:
            e = e + rng.normal(0.0, DRIFT_SD)
        out.append(e)
    return np.concatenate(out)


def fit(op, M, y):
    resid = lambda th: M @ (apply_op(np.exp(th[0]), np.exp(th[1]), op) - y) / SIG_N
    r = least_squares(resid, np.log([D_INIT, K_INIT]), diff_step=1e-4, bounds=BOUNDS)
    cov = np.linalg.inv(r.jac.T @ r.jac)
    lD, lk = r.x
    sd_lam = 0.5 * np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    lam = np.sqrt(np.exp(lD) / np.exp(lk))
    return {'D_err': 100 * (np.exp(lD) - D_TRUE) / D_TRUE,
            'k_err': 100 * (np.exp(lk) - K_TRUE) / K_TRUE,
            'lam_err': 100 * (lam - LAM_TRUE) / LAM_TRUE,
            'covers_D': bool(abs(lD - np.log(D_TRUE)) <= 1.96 * np.sqrt(cov[0, 0])),
            'covers_k': bool(abs(lk - np.log(K_TRUE)) <= 1.96 * np.sqrt(cov[1, 1])),
            'covers_lam': bool(abs(np.log(lam / LAM_TRUE)) <= 1.96 * sd_lam),
            'at_bound': bool(np.any(np.isclose(r.x, BOUNDS[0], atol=1e-6) |
                                    np.isclose(r.x, BOUNDS[1], atol=1e-6)))}


def summarize(fits):
    out = {}
    for q in ('D', 'k', 'lam'):
        v = np.array([f[f'{q}_err'] for f in fits])
        out[f'median_{q}'] = float(np.median(v))
        out[f'rmse_{q}'] = float(np.sqrt(np.mean(v ** 2)))
        out[f'covered_{q}'] = int(sum(f[f'covers_{q}'] for f in fits))
    out['n_at_bound'] = int(sum(f['at_bound'] for f in fits))
    out['n'] = len(fits)
    return out


def main():
    ref_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    stored = json.loads(ref_path.read_text())['scenarios'] if ref_path else None
    res = {'true_tau_ms': MEAS_TAU, 'tau_fit_ms': TAU_FITS, 'seeds': [SEEDS[0], SEEDS[-1]],
           'scenarios': {}}
    for name in SCENARIOS:
        sc = SCEN[name]
        clean = apply_op(D_TRUE, K_TRUE, build_operator(sc['a'], sc['tau'], TIMES))
        M = block_matrix(sc['rho_tau'], sc['drift'], TIMES)
        ys = {ns: clean + make_noise(sc['rho_tau'], sc['drift'], ns) for ns in SEEDS}
        res['scenarios'][name] = {}
        for tf in TAU_FITS:
            op = build_fit_operator(sc['a'], tf)
            fits = {str(ns): fit(op, M, ys[ns]) for ns in SEEDS}
            summ = summarize(list(fits.values()))
            if stored is not None and tf == MEAS_TAU:
                ref = stored[name]['ref']
                summ['max_abs_diff_vs_stored'] = {
                    q: float(max(abs(fits[s][f'{q}_err'] - ref[s]['aware'][f'{q}_err'])
                                 for s in fits)) for q in ('D', 'k')}
            res['scenarios'][name][str(tf)] = {'summary': summ, 'fits': fits}
            s = summ
            print(f"{name:<8} tau_fit {tf:3.1f} | median D {s['median_D']:+6.1f} k {s['median_k']:+6.1f} "
                  f"lam {s['median_lam']:+6.1f} | RMSE D {s['rmse_D']:5.1f} k {s['rmse_k']:5.1f} "
                  f"lam {s['rmse_lam']:5.1f} | cover D {s['covered_D']}/50 k {s['covered_k']}/50 "
                  f"lam {s['covered_lam']}/50 | at bound {s['n_at_bound']}"
                  + (f" | diff vs stored {s['max_abs_diff_vs_stored']}" if 'max_abs_diff_vs_stored' in s else ''),
                  flush=True)
    if out_path:
        out_path.write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
