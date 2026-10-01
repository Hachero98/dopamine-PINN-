"""Synthetic observation designs.

All designs return a dict with arrays x_d, y_d, t_d (locations, um and ms), C_d (noisy
concentrations, uM) and C_clean (noise-free values). Random draws follow the order of
the original notebooks (locations, then times, then noise), so a given seed gives the
same locations and noise as in the paper.
"""
import numpy as np
import jax.numpy as jnp

from . import config as cfg
from .solver import interp_frames


def _sample(frames, solver, x, y, t):
    return np.asarray(interp_frames(frames, solver.dx, jnp.asarray(x), jnp.asarray(y),
                                    jnp.asarray(t), solver.dt_store, solver.L))


def scattered(frames, solver, seed, n_obs=cfg.N_OBS, noise_sd=cfg.NOISE_SD,
              t_window=cfg.T_WINDOW, valid=None, L=cfg.L):
    """Points uniform over the square and the time window, Gaussian noise of SD noise_sd.

    valid(x, y) -> bool restricts locations (rejected points are redrawn in blocks).
    noise_sd = 0 still consumes the noise draws, so all noise levels share locations
    and the standardized noise (as in the paper's noise sweep).
    """
    rng = np.random.default_rng(seed)
    if valid is None:
        ox = rng.uniform(-L / 2, L / 2, n_obs)
        oy = rng.uniform(-L / 2, L / 2, n_obs)
    else:
        ox, oy = np.empty(0), np.empty(0)
        while ox.size < n_obs:
            cx = rng.uniform(-L / 2, L / 2, 4 * n_obs)
            cy = rng.uniform(-L / 2, L / 2, 4 * n_obs)
            ok = valid(cx, cy)
            ox, oy = np.concatenate([ox, cx[ok]]), np.concatenate([oy, cy[ok]])
        ox, oy = ox[:n_obs], oy[:n_obs]
    ot = rng.uniform(*t_window, n_obs)
    clean = _sample(frames, solver, ox, oy, ot)
    return {'x_d': ox, 'y_d': oy, 't_d': ot, 'C_d': clean + rng.normal(0.0, noise_sd, n_obs),
            'C_clean': clean}


def grid_nodes(frames, solver, seed, n_obs=cfg.N_OBS, noise_sd=cfg.NOISE_SD, t_window=cfg.T_WINDOW):
    """Points at random extracellular grid nodes (exact in space), uniform times.

    Used with obstacles, where interpolation across a staircase boundary would mix in
    solid nodes.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-solver.L / 2, solver.L / 2, solver.nx)
    fluid = np.argwhere(solver.mask > 0)
    sel = fluid[rng.choice(len(fluid), n_obs, replace=False)]
    ox, oy = x[sel[:, 0]], x[sel[:, 1]]
    ot = rng.uniform(*t_window, n_obs)
    clean = _sample(frames, solver, ox, oy, ot)
    return {'x_d': ox, 'y_d': oy, 't_d': ot, 'C_d': clean + rng.normal(0, noise_sd, n_obs),
            'C_clean': clean}


def electrode(frames, solver, sites, dt, seed, noise_sd=cfg.NOISE_SD, t_window=cfg.T_WINDOW):
    """Dense time series at fixed sites [(x, y), ...], every dt ms (FSCV-like design)."""
    t_s = np.arange(t_window[0], t_window[1] + 1e-9, dt)
    xs = np.concatenate([np.full_like(t_s, sx) for sx, _ in sites])
    ys = np.concatenate([np.full_like(t_s, sy) for _, sy in sites])
    ts = np.concatenate([t_s for _ in sites])
    clean = _sample(frames, solver, xs, ys, ts)
    return {'x_d': xs, 'y_d': ys, 't_d': ts,
            'C_d': clean + np.random.default_rng(seed).normal(0.0, noise_sd, ts.size), 'C_clean': clean}


def merge(*designs):
    return {k: np.concatenate([d[k] for d in designs]) for k in designs[0]}


def subset(obs, idx):
    return {k: np.asarray(v)[idx] for k, v in obs.items()}


ELECTRODE_DESIGNS = {
    'S1_r1.0': dict(sites=[(1.0, 0.0)], dt=0.5),
    'S1_r2.0': dict(sites=[(2.0, 0.0)], dt=0.5),
    'S1_r2.5': dict(sites=[(2.5, 0.0)], dt=0.5),
    'S3': dict(sites=[(1.0, 0.0), (0.0, 2.0), (-2.5 / np.sqrt(2), -2.5 / np.sqrt(2))], dt=0.5),
    'S1_r1.0_coarse': dict(sites=[(1.0, 0.0)], dt=5.0),
}
