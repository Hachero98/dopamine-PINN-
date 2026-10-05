"""Experimental-design map for recovering D and k from electrode-like recordings.

For an observation design (sites, sampling times) and Gaussian noise of SD
sigma = eta * C0, the Fisher information of (log D, log k) at the true
parameters is F = J^T J / sigma^2, with J the sensitivities of the observed
concentrations. Its inverse is the Cramer-Rao bound (CRB) on the covariance of
any unbiased estimator; sqrt(diag) in log space is the best achievable
relative standard deviation of D and k.

Sensitivities:
  d C / d log k = -k t C           exact, because C = exp(-k t) u(x, y, t; D)
  d C / d log D  central difference of two FD solves at D exp(+-h)
All solves share one time step so the difference is not polluted by a change
of discretization. The FD solver is the one of dopamine_PINN.ipynb.

Outputs (in this folder): design_map.json and design_map_overview.png.
Part 2 checks the CRB against the actual FD reference estimator (non-linear
least squares with the same solver) over many noise realizations.

Usage:  python experimental_design_map.py [n_monte_carlo]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
D_TRUE, K_TRUE = 0.32, 0.020          # um^2/ms, 1/ms
L, T, SIGMA, C0 = 5.0, 50.0, 0.5, 1.0
ETA = 0.02                            # noise SD as a fraction of C0
T_MIN, T_MAX = 0.5, 0.9 * T           # observation window of the paper
N_MC = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def fd_reference(D=D_TRUE, k=K_TRUE, nx=81, ny=81, nt=None, dt_store=0.02):
    """Explicit 2D FD solver with zero-flux walls (as in dopamine_PINN.ipynb)."""
    dx = L / (nx - 1); dy = L / (ny - 1)
    if nt is None:
        dt_max = 0.95 / (2.0 * D * (1.0 / dx**2 + 1.0 / dy**2))
        nt = int(np.ceil(T / dt_max)) + 1
    dt = T / (nt - 1)
    assert dt <= 1.0 / (2.0 * D * (1.0 / dx**2 + 1.0 / dy**2)), 'FD stability violated.'
    stride = max(1, int(round(dt_store / dt)))
    x = np.linspace(-L / 2, L / 2, nx); y = np.linspace(-L / 2, L / 2, ny)
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    C = C0 * np.exp(-(Xg**2 + Yg**2) / (2.0 * SIGMA**2))
    frames, t_frames = [C.copy()], [0.0]
    for n in range(1, nt):
        lap = np.zeros_like(C)
        lap[1:-1, :] += (C[2:, :] - 2 * C[1:-1, :] + C[:-2, :]) / dx**2
        lap[0, :] += 2 * (C[1, :] - C[0, :]) / dx**2
        lap[-1, :] += 2 * (C[-2, :] - C[-1, :]) / dx**2
        lap[:, 1:-1] += (C[:, 2:] - 2 * C[:, 1:-1] + C[:, :-2]) / dy**2
        lap[:, 0] += 2 * (C[:, 1] - C[:, 0]) / dy**2
        lap[:, -1] += 2 * (C[:, -2] - C[:, -1]) / dy**2
        C = C + D * dt * lap - k * dt * C
        if n % stride == 0 or n == nt - 1:
            frames.append(C.copy()); t_frames.append(n * dt)
    return x, y, np.array(t_frames), np.stack(frames)


def interp(sol):
    x, y, t, H = sol
    return RegularGridInterpolator((t, x, y), H, bounds_error=False, fill_value=None)


# ---- three solves on a common time step ------------------------------------------
H_STEP = 0.02
dx = L / 80
NT = int(np.ceil(T / (0.95 / (2.0 * D_TRUE * np.exp(H_STEP) * (2.0 / dx**2))))) + 1
t0 = time.time()
f0 = interp(fd_reference(D_TRUE, K_TRUE, nt=NT))
fp = interp(fd_reference(D_TRUE * np.exp(H_STEP), K_TRUE, nt=NT))
fm = interp(fd_reference(D_TRUE * np.exp(-H_STEP), K_TRUE, nt=NT))
print(f'3 FD solves in {time.time() - t0:.1f} s')


def crb(xs, ys, ts, eta=ETA):
    """Relative SDs of D and k (CRB, in %) and their correlation."""
    P = np.stack([ts, xs, ys], axis=1)
    c = f0(P)
    jD = (fp(P) - fm(P)) / (2 * H_STEP)
    jk = -K_TRUE * ts * c
    J = np.stack([jD, jk], axis=1)
    F = J.T @ J / (eta * C0) ** 2
    cov = np.linalg.inv(F)
    sd = np.sqrt(np.diag(cov))
    return 100 * sd[0], 100 * sd[1], cov[0, 1] / (sd[0] * sd[1])


def single_site(x, y, dt, tmax=T_MAX):
    ts = np.arange(T_MIN, tmax + 1e-9, dt)
    return np.full_like(ts, x), np.full_like(ts, y), ts


def multi_site(sites, dt, tmax=T_MAX):
    parts = [single_site(x, y, dt, tmax) for x, y in sites]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


out = {'D_true': D_TRUE, 'k_true': K_TRUE, 'eta': ETA, 'window_ms': [T_MIN, T_MAX],
       'note': 'CRB relative SDs in percent; correlation rho of log D and log k'}

# ---- reference: the scattered design of the paper (400 points, seed 4001) ---------
rng = np.random.default_rng(4001)
sx, sy = rng.uniform(-L / 2, L / 2, 400), rng.uniform(-L / 2, L / 2, 400)
st = rng.uniform(T_MIN, T_MAX, 400)
sD, sk, srho = crb(sx, sy, st)
out['scattered_400'] = {'sd_D': sD, 'sd_k': sk, 'rho': srho}
print(f'scattered 400 points: SD D {sD:.1f}%, SD k {sk:.1f}%, rho {srho:+.2f}')

# ---- map A: single site, distance x sampling interval ------------------------------
R = np.round(np.linspace(0.0, 2.5, 26), 3)
DT = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
mapA = {'r_um': R.tolist(), 'dt_ms': DT, 'sd_D': [], 'sd_k': [], 'rho': []}
for dtv in DT:
    rowD, rowk, rowr = [], [], []
    for r in R:
        d, k_, rho = crb(*single_site(r, 0.0, dtv))
        rowD.append(d); rowk.append(k_); rowr.append(rho)
    mapA['sd_D'].append(rowD); mapA['sd_k'].append(rowk); mapA['rho'].append(rowr)
out['single_site_axis'] = mapA
# along the diagonal, out to the corner (r up to 2.5 * sqrt 2)
Rd = np.round(np.linspace(0.0, 2.5 * np.sqrt(2), 21), 3)
out['single_site_diagonal_dt0.5'] = {
    'r_um': Rd.tolist(),
    'sd_D': [crb(*single_site(r / np.sqrt(2), r / np.sqrt(2), 0.5))[0] for r in Rd],
    'sd_k': [crb(*single_site(r / np.sqrt(2), r / np.sqrt(2), 0.5))[1] for r in Rd]}

# ---- map B: recording length (k T effect), single site r = 1 um, dt 0.5 ms --------
TM = [5, 10, 20, 30, 45]
out['recording_length_r1_dt0.5'] = {
    't_max_ms': TM,
    'sd_D': [crb(*single_site(1.0, 0.0, 0.5, tm))[0] for tm in TM],
    'sd_k': [crb(*single_site(1.0, 0.0, 0.5, tm))[1] for tm in TM]}

# ---- map C: number of sites --------------------------------------------------------
site_sets = {
    '1 site (r=1)': [(1.0, 0.0)],
    '2 sites (r=0.5, 2)': [(0.5, 0.0), (0.0, 2.0)],
    '3 sites (r=1, 2, 2.5)': [(1.0, 0.0), (0.0, 2.0), (-2.5 / np.sqrt(2), -2.5 / np.sqrt(2))],
    '4 sites (r=0.5, 1, 1.5, 2)': [(0.5, 0.0), (0.0, 1.0), (-1.5, 0.0), (0.0, -2.0)],
}
out['multi_site_dt0.5'] = {}
for name, sites in site_sets.items():
    d, k_, rho = crb(*multi_site(sites, 0.5))
    out['multi_site_dt0.5'][name] = {'sites': sites, 'sd_D': d, 'sd_k': k_, 'rho': rho}
    print(f'{name:<28} SD D {d:5.1f}%  SD k {k_:5.1f}%  rho {rho:+.2f}')

# ---- part 2: does the CRB match the actual reference estimator? --------------------
def fd_predict(D, k, xs, ys, ts):
    return interp(fd_reference(D, k))(np.stack([ts, xs, ys], axis=1))


checks = {'S1 r=1 dt=0.5': single_site(1.0, 0.0, 0.5),
          'S1 r=2 dt=0.5': single_site(2.0, 0.0, 0.5),
          'S3 dt=0.5': multi_site(site_sets['3 sites (r=1, 2, 2.5)'], 0.5)}
out['monte_carlo'] = {'n_draws': N_MC}
for name, (xs, ys, ts) in checks.items():
    clean = f0(np.stack([ts, xs, ys], axis=1))
    est = []
    t1 = time.time()
    for draw in range(N_MC):
        noisy = clean + np.random.default_rng(10_000 + draw).normal(0, ETA * C0, ts.size)
        res = least_squares(lambda th: fd_predict(np.exp(th[0]), np.exp(th[1]), xs, ys, ts) - noisy,
                            np.log([0.30, 0.016]), diff_step=1e-3,
                            bounds=(np.log([0.02, 1e-4]), np.log([5.0, 0.5])))
        est.append(np.exp(res.x))
    est = np.array(est)
    eD = 100 * (est[:, 0] - D_TRUE) / D_TRUE
    ek = 100 * (est[:, 1] - K_TRUE) / K_TRUE
    d, k_, rho = crb(xs, ys, ts)
    out['monte_carlo'][name] = {
        'crb_sd_D': d, 'crb_sd_k': k_,
        'empirical_sd_D': float(eD.std(ddof=1)), 'empirical_sd_k': float(ek.std(ddof=1)),
        'mean_err_D': float(eD.mean()), 'mean_err_k': float(ek.mean()),
        'errors_D': eD.tolist(), 'errors_k': ek.tolist()}
    print(f'MC {name:<16} D: CRB {d:5.1f}% vs empirical SD {eD.std(ddof=1):5.1f}% (mean {eD.mean():+.1f}%) | '
          f'k: CRB {k_:5.1f}% vs {ek.std(ddof=1):5.1f}% (mean {ek.mean():+.1f}%)  [{time.time()-t1:.0f} s]')
    (HERE / 'design_map.json').write_text(json.dumps(out, indent=2))

(HERE / 'design_map.json').write_text(json.dumps(out, indent=2))
print('saved design_map.json')
