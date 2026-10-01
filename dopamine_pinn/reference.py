"""Reference estimator: least squares through the FD solver, with a Laplace posterior.

The estimator fits the parameters with the same bounded-domain solver that generated
the data, so its error reflects only the noise and the observation design; it is the
benchmark the PINN is compared with, not a competing method. Jacobians are exact
(forward-mode autodiff through the solver).

Also here: the Levenberg-Marquardt baseline of the paper, which fits the closed-form
infinite-domain solution.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
from scipy.stats import chi2

from . import config as cfg
from .solver import gaussian_ic, interp_frames, linear_sink, mm_sink, analytical_infinite


def ref_fit(predict, theta0, C_obs, sigma, lb, ub):
    """Bounded non-linear least squares; returns estimate, Laplace covariance and fit statistics.

    predict(theta) -> model concentrations at the observation points (JAX).
    sigma: noise SD; used for the covariance (Gauss-Newton Hessian, flat prior) and chi^2.
    """
    f = jax.jit(predict)
    J = jax.jit(jax.jacfwd(predict))
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    th0 = np.clip(np.asarray(theta0, float), lb + 1e-9, ub - 1e-9)
    r = least_squares(lambda th: np.asarray(f(jnp.asarray(th))) - C_obs, th0,
                      jac=lambda th: np.asarray(J(jnp.asarray(th))),
                      bounds=(lb, ub), x_scale='jac', xtol=1e-10, ftol=1e-12, gtol=1e-10)
    cov = np.linalg.pinv(r.jac.T @ r.jac) * sigma**2
    n, p = C_obs.size, len(th0)
    rss = float(np.sum(r.fun**2))
    return {'theta': r.x, 'cov': cov, 'rms': float(np.sqrt(rss / n)),
            'chi2': rss / sigma**2, 'dof': n - p, 'p_chi2': float(chi2.sf(rss / sigma**2, n - p)),
            'aic': float(n * np.log(rss / n) + 2 * p), 'nfev': int(r.nfev),
            'at_bound': bool(np.any(np.isclose(r.x, lb) | np.isclose(r.x, ub)))}


def _obs_jax(obs):
    return tuple(jnp.asarray(obs[k]) for k in ('x_d', 'y_d', 't_d'))


def linear_predict(obs, solver, amp=cfg.C0):
    """theta = (log D, log k) -> FD concentrations at the observation points."""
    x, y, t = _obs_jax(obs)
    C_init = gaussian_ic(solver.nx, amp, L=solver.L)

    def predict(th):
        fr = solver(jnp.exp(th[0]), linear_sink(jnp.exp(th[1])), C_init)
        return interp_frames(fr, solver.dx, x, y, t, solver.dt_store, solver.L)
    return predict


def mm_predict(obs, solver, amp=cfg.C0):
    """theta = (log D, log Vmax, log Km) -> FD concentrations at the observation points."""
    x, y, t = _obs_jax(obs)
    C_init = gaussian_ic(solver.nx, amp, L=solver.L)

    def predict(th):
        fr = solver(jnp.exp(th[0]), mm_sink(jnp.exp(th[1]), jnp.exp(th[2])), C_init)
        return interp_frames(fr, solver.dx, x, y, t, solver.dt_store, solver.L)
    return predict


def fit_linear(obs, solver, noise_sd=cfg.NOISE_SD, amp=cfg.C0, D0=cfg.D_INIT, k0=cfg.K_INIT):
    """Reference fit of (D, k). Returns D, k, Laplace SDs of log D / log k and the full result."""
    r = ref_fit(linear_predict(obs, solver, amp), np.log([D0, k0]), np.asarray(obs['C_d']), noise_sd,
                np.log([0.02, 1e-4]), np.log([0.97 * solver.D_cap, 0.5]))
    sd = np.sqrt(np.diag(r['cov']))
    return {'D': float(np.exp(r['theta'][0])), 'k': float(np.exp(r['theta'][1])),
            'sd_logD': float(sd[0]), 'sd_logk': float(sd[1]),
            'rho': float(r['cov'][0, 1] / (sd[0] * sd[1])), 'fit': r}


def fit_mm(obs, solver, noise_sd=cfg.NOISE_SD, amp=cfg.C0,
           theta0=(cfg.D_INIT, 0.8 * cfg.VMAX_TRUE, 1.2 * cfg.KM_TRUE)):
    r = ref_fit(mm_predict(obs, solver, amp), np.log(theta0), np.asarray(obs['C_d']), noise_sd,
                np.log([0.02, 1e-5, 1e-3]), np.log([0.97 * solver.D_cap, 0.5, 50.0]))
    sd = np.sqrt(np.diag(r['cov']))
    D, Vm, Km = np.exp(r['theta'])
    return {'D': float(D), 'Vmax': float(Vm), 'Km': float(Km), 'sd_log': sd.tolist(),
            'corr': (r['cov'] / np.outer(sd, sd)).tolist(), 'fit': r}


def laplace_interval(value, sd_log, z=1.96):
    """95% interval of a positive parameter from the Laplace SD of its logarithm."""
    return value * np.exp(-z * sd_log), value * np.exp(z * sd_log)


def lm_analytical(obs, D0=cfg.D_INIT, k0=cfg.K_INIT, sigma=cfg.SIGMA, C0=cfg.C0):
    """The paper's classical baseline: LM fit of the infinite-domain Gaussian solution."""
    x, y, t, C = (np.asarray(obs[k]) for k in ('x_d', 'y_d', 't_d', 'C_d'))
    r = least_squares(lambda th: analytical_infinite(x, y, t, th[0], th[1], sigma, C0) - C,
                      np.array([D0, k0]), method='lm')
    return {'D': float(r.x[0]), 'k': float(r.x[1]), 'cost': float(r.cost), 'nfev': int(r.nfev)}
