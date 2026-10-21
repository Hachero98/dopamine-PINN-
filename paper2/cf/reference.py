"""Reference estimator and the Cramer-Rao bounds, for any mechanism.

Reference: least squares through the FD model that generated the data, Laplace
covariance on the log scale. It runs the same stages as the PINN. Its error reflects
only the noise and the design, so it is the yardstick, not a competing method.

Bound: Fisher information F = J^T J / sigma^2 at the truth; for an effect g(theta),
Var(g_hat) >= grad g^T F^-1 grad g (paper 2, T2).
"""
import numpy as np
import jax

from dopamine_pinn.reference import ref_fit
from .model import untreated


def fit(fwd, obs, free=None, fixed=None, init=None):
    mech = fwd.mech
    free = list(mech.params if free is None else free)
    fixed = dict(fixed or {})
    init = {**mech.init, **(init or {})}
    lo = {**mech.BOUNDS, 'D': (mech.BOUNDS['D'][0], 0.97 * fwd.solver.D_cap)}
    lb = np.log([lo[p][0] for p in free])
    ub = np.log([lo[p][1] for p in free])
    r = ref_fit(fwd.predictor(obs, free, fixed), np.log([init[p] for p in free]), np.asarray(obs['C_d']),
                fwd.design.noise_sd, lb, ub)
    sd = np.sqrt(np.clip(np.diag(r['cov']), 0, None))
    params = {**fixed, **{p: float(np.exp(r['theta'][i])) for i, p in enumerate(free)}}
    return {'params': params, 'free': free, 'sd_log': dict(zip(free, sd.tolist())), 'cov_log': r['cov'].tolist(),
            'corr': (r['cov'] / np.outer(sd, sd)).tolist(), 'chi2_p': r['p_chi2'], 'at_bound': r['at_bound']}


def staged(fwd, obs):
    """Stages 2-4, as the PINN: baseline from untreated data, then the drug parameter with
    the baseline pinned, then everything."""
    mech = fwd.mech
    s2 = fit(fwd, untreated(obs), free=mech.baseline_params, fixed={mech.dose_param: mech.init[mech.dose_param]})
    s3 = fit(fwd, obs, free=[mech.dose_param], fixed={p: s2['params'][p] for p in mech.baseline_params})
    s4 = fit(fwd, obs, init=s3['params'])
    return {'stage2': s2, 'stage3': s3, 'stage4': s4}


def fisher(fwd, obs, theta=None):
    theta = fwd.theta_true if theta is None else theta
    J = np.asarray(jax.jacfwd(fwd.predictor(obs))(theta))
    return J.T @ J / fwd.design.noise_sd**2


def inverse(F):
    """F^-1, with the condition number (large: some parameters not separately identifiable)."""
    cond = float(np.linalg.cond(F))
    return (np.linalg.pinv(F, rcond=1e-15) if cond > 1e14 else np.linalg.inv(F)), cond


def delta_sd(cov, grad):
    """SD of a scalar (grad (p,)) or of each entry of an array (..., p)."""
    return np.sqrt(np.clip(np.einsum('...i,ij,...j->...', grad, cov, grad), 0, None))
