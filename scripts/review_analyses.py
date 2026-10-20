"""Analyses added in response to the pre-submission review (BSPC manuscript).

1. Cramer-Rao bound for each sensor scenario of the two-site design (Table 3),
   with the generating sensor operator and the whitening/offset projection, and
   for the offset scenario without the pre-stimulus baseline.
2. Reference estimator over 50 noise realizations (seeds 4001-4050) for every
   electrode design of Table 2, with the exact image-series solution (Appendix A)
   as both the truth and the fitted model. Seeds 4001-4005 reproduce the noise
   of the original finite-difference runs exactly.
3. Cramer-Rao bound of each electrode design, as a check of Table 2.

NumPy/SciPy only, about a minute on a laptop.
Usage:  python scripts/review_analyses.py <electrode_design.json> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

# Model and noise (as in the notebooks)
D_TRUE, K_TRUE, L, SIGMA, C0 = 0.32, 0.020, 5.0, 0.5, 1.0
D_INIT, K_INIT = 0.30, 0.016
ETA = 0.02
SIG_N = ETA * C0
BOUNDS = (np.log([0.02, 1e-4]), np.log([5.0, 0.5]))

# Sensor experiment (dopamine_PINN_tuning.ipynb, RUN_MEAS cell)
MEAS_A, MEAS_TAU, MEAS_TAU_N = 0.5, 2.0, 2.0
MEAS_DT, MEAS_TMIN, MEAS_TMAX, MEAS_PRE = 0.5, 0.5, 45.0, 10.0
MEAS_SITES = [(0.5, 0.0), (0.0, 2.0)]
SCEN = {
    'M0_ideal':  dict(a=0.0,    tau=0.0,      rho_tau=0.0,        drift=False),
    'M1_tip':    dict(a=MEAS_A, tau=0.0,      rho_tau=0.0,        drift=False),
    'M2_lag':    dict(a=0.0,    tau=MEAS_TAU, rho_tau=0.0,        drift=False),
    'M3_colour': dict(a=0.0,    tau=0.0,      rho_tau=MEAS_TAU_N, drift=False),
    'M4_drift':  dict(a=0.0,    tau=0.0,      rho_tau=0.0,        drift=True),
    'M5_all':    dict(a=MEAS_A, tau=MEAS_TAU, rho_tau=MEAS_TAU_N, drift=True),
}


def exact_C(D, k, x, y, t):
    x, y, t = np.asarray(x, float), np.asarray(y, float), np.asarray(t, float)
    v = SIGMA**2 + 2.0 * D * t
    m = np.arange(-6, 7).reshape((-1,) + (1,) * x.ndim)
    gx = np.exp(-(x[None] - m * L) ** 2 / (2.0 * v[None])).sum(0)
    gy = np.exp(-(y[None] - m * L) ** 2 / (2.0 * v[None])).sum(0)
    return C0 * SIGMA**2 / v * np.exp(-k * t) * gx * gy


def disk_rule(a):          # fine rule used to generate the data
    if a <= 0:
        return np.zeros((1, 2)), np.ones(1)
    gu, gw = np.polynomial.legendre.leggauss(4)
    r = a * np.sqrt((gu + 1) / 2)
    th = 2 * np.pi * np.arange(16) / 16
    off = np.array([(ri * np.cos(t_), ri * np.sin(t_)) for ri in r for t_ in th])
    w = np.array([wi / 2 / 16 for wi in gw for _ in th])
    return off, w


def lag_rule(tau):         # fine rule used to generate the data
    if tau <= 0:
        return np.zeros(1), np.ones(1)
    s = np.arange(0.0, 12 * tau + 1e-9, 0.02)
    w = np.exp(-s / tau) / tau * 0.02
    w[0] *= 0.5; w[-1] *= 0.5
    return s, w * (1 - np.exp(-s[-1] / tau)) / w.sum()


def build_operator(a, tau, times, sites=MEAS_SITES):
    off, wd = disk_rule(a)
    s, wl = lag_rule(tau)
    X, Y, Tn, W = [], [], [], []
    for sx, sy in sites:
        for tt in times:
            tq = tt - s
            X.append(np.repeat(sx + off[:, 0], s.size))
            Y.append(np.repeat(sy + off[:, 1], s.size))
            Tn.append(np.tile(np.maximum(tq, 0.0), off.shape[0]))
            W.append(np.outer(wd, wl * (tq >= 0)).ravel())
    return np.array(X), np.array(Y), np.array(Tn), np.array(W)


def apply_op(D, k, op):
    X, Y, Tn, W = op
    return (W * exact_C(D, k, X, Y, Tn)).sum(1)


def site_matrix(rho_tau, drift, times):
    n = times.size
    Wm = np.eye(n)
    if rho_tau > 0:
        rho = np.exp(-np.diff(times) / rho_tau)
        for i in range(1, n):
            Wm[i, i] = 1.0 / np.sqrt(1 - rho[i - 1] ** 2)
            Wm[i, i - 1] = -rho[i - 1] / np.sqrt(1 - rho[i - 1] ** 2)
    if drift:
        B = Wm @ np.ones((n, 1))
        Wm = (np.eye(n) - B @ np.linalg.pinv(B)) @ Wm
    return Wm


def block_matrix(rho_tau, drift, times, n_sites=len(MEAS_SITES)):
    Ms = site_matrix(rho_tau, drift, times)
    n = times.size
    M = np.zeros((n_sites * n, n_sites * n))
    for i in range(n_sites):
        M[i * n:(i + 1) * n, i * n:(i + 1) * n] = Ms
    return M


def crb_pct(f, M=None, h=1e-5):
    """Relative CRB (%) of D and k for noise-free output f(logD, logk)."""
    th = np.log([D_TRUE, K_TRUE])
    J = np.empty((f(*th).size, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h
        J[:, j] = (f(*(th + e)) - f(*(th - e))) / (2 * h)
    if M is not None:
        J = M @ J
    F = J.T @ J / SIG_N**2
    return (100 * np.sqrt(np.diag(np.linalg.inv(F)))).tolist()


def sensor_bounds():
    out = {}
    for pre in (True, False):
        times = (np.r_[np.arange(-MEAS_PRE, 0.0, MEAS_DT), np.arange(MEAS_TMIN, MEAS_TMAX + 1e-9, MEAS_DT)]
                 if pre else np.arange(MEAS_TMIN, MEAS_TMAX + 1e-9, MEAS_DT))
        for name, sc in SCEN.items():
            if not pre and name not in ('M0_ideal', 'M4_drift', 'M5_all'):
                continue
            op = build_operator(sc['a'], sc['tau'], times)
            M = block_matrix(sc['rho_tau'], sc['drift'], times)
            f = lambda lD, lk: apply_op(np.exp(lD), np.exp(lk), op)
            out[name + ('' if pre else '_no_baseline')] = crb_pct(f, M)
    return out


def electrode_fit(xs, ys, ts, y):
    resid = lambda th: exact_C(np.exp(th[0]), np.exp(th[1]), xs, ys, ts) - y
    r = least_squares(resid, np.log([D_INIT, K_INIT]), diff_step=1e-4, bounds=BOUNDS)
    cov = np.linalg.inv(r.jac.T @ r.jac) * SIG_N**2
    sd = np.sqrt(np.diag(cov))
    D_, k_ = np.exp(r.x)
    return {'D_err': 100 * (D_ - D_TRUE) / D_TRUE, 'k_err': 100 * (k_ - K_TRUE) / K_TRUE,
            'covers_D': bool(abs(np.log(D_ / D_TRUE)) <= 1.96 * sd[0]),
            'covers_k': bool(abs(np.log(k_ / K_TRUE)) <= 1.96 * sd[1]),
            'at_bound': bool(np.any(np.isclose(r.x, BOUNDS[0], atol=1e-6) |
                                    np.isclose(r.x, BOUNDS[1], atol=1e-6)))}


def electrode_designs(el, seeds):
    out = {}
    for des in el['designs']:
        t_s = np.arange(0.5, 45.0 + 1e-9, des['dt'])
        xs = np.concatenate([np.full_like(t_s, sx) for sx, sy in des['sites']])
        ys = np.concatenate([np.full_like(t_s, sy) for sx, sy in des['sites']])
        ts = np.concatenate([t_s for _ in des['sites']])
        clean = exact_C(D_TRUE, K_TRUE, xs, ys, ts)
        fits = {}
        for ns in seeds:
            noisy = clean + np.random.default_rng(ns).normal(0.0, SIG_N, ts.size)
            fits[str(ns)] = electrode_fit(xs, ys, ts, noisy)
        Dv = np.array([f['D_err'] for f in fits.values()])
        kv = np.array([f['k_err'] for f in fits.values()])
        f_bound = lambda lD, lk: exact_C(np.exp(lD), np.exp(lk), xs, ys, ts)
        out[des['name']] = {
            'sites': des['sites'], 'dt': des['dt'], 'n_obs': int(ts.size), 'n_draws': len(seeds),
            'crb_pct': crb_pct(f_bound),
            'rmse_D': float(np.sqrt(np.mean(Dv**2))), 'rmse_k': float(np.sqrt(np.mean(kv**2))),
            'median_D': float(np.median(Dv)), 'median_k': float(np.median(kv)),
            'covered_D': int(sum(f['covers_D'] for f in fits.values())),
            'covered_k': int(sum(f['covers_k'] for f in fits.values())),
            'n_at_bound': int(sum(f['at_bound'] for f in fits.values())),
            'fits': fits}
        # the original finite-difference reference on seeds 4001-4005, for comparison
        fd5 = {str(d['noise_seed']): d['reference'] for d in des['draws']}
        out[des['name']]['max_abs_diff_vs_fd_first5'] = {
            q: float(max(abs(fits[s][f'{q}_err'] - fd5[s][f'{q}_err_signed_pct']) for s in fd5))
            for q in ('D', 'k')}
    return out


if __name__ == '__main__':
    el = json.loads(Path(sys.argv[1]).read_text())
    res = {'sensor_crb_pct': sensor_bounds(),
           'electrode': electrode_designs(el, list(range(4001, 4051)))}
    Path(sys.argv[2]).write_text(json.dumps(res, indent=2))
    print('sensor CRB (D, k) %:')
    for n, v in res['sensor_crb_pct'].items():
        print(f'  {n:<24} {v[0]:6.1f} {v[1]:6.1f}')
    print('electrode designs, 50 realizations:')
    for n, d in res['electrode'].items():
        print(f"  {n:<13} CRB {d['crb_pct'][0]:5.1f} {d['crb_pct'][1]:5.1f} | RMSE {d['rmse_D']:5.1f} "
              f"{d['rmse_k']:5.1f} | median {d['median_D']:+5.1f} {d['median_k']:+5.1f} | cover "
              f"{d['covered_D']}/{d['n_draws']} {d['covered_k']}/{d['n_draws']} | at bound {d['n_at_bound']}"
              f" | max diff vs FD (5) {d['max_abs_diff_vs_fd_first5']['D']:.2f} "
              f"{d['max_abs_diff_vs_fd_first5']['k']:.2f}")
