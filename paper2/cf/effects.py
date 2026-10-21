"""Counterfactual effects of do(A = a) on the evaluation grid, and their bounds.

C^(a)   potential outcome (concentration field under dose a)
tau_a   = C^(a) - C^(0), the pointwise effect
ATE_a(t) domain average of tau_a;  CE_a(x, y) time integral;  GTE_a time average of ATE_a.

The grid is the FD nodes every eval_dt ms on [0, T]; domain averages use trapezoid
weights on the square (normalised to 1). These are mechanistic counterfactual effects,
valid within the assumed PDE and drug pathway.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .reference import inverse, delta_sd


class EffectGrid:
    def __init__(self, fwd, eval_dt=1.0):
        s = fwd.solver
        self.fwd = fwd
        self.ts = np.arange(0.0, s.T_end + 1e-9, eval_dt)
        self.fidx = np.round(self.ts / s.dt_store).astype(int)
        self.x = np.linspace(-s.L / 2, s.L / 2, s.nx)
        w = np.ones(s.nx)
        w[0] = w[-1] = 0.5
        self.W = np.outer(w, w) * s.mask / np.sum(np.outer(w, w) * s.mask)

    # JAX versions (differentiable in theta)
    def field(self, theta, a):
        return self.fwd.frames(theta, a)[self.fidx]

    def tau(self, theta, a):
        return self.field(theta, a) - self.field(theta, 0.0)

    def ate(self, theta, a):
        return jnp.sum(self.tau(theta, a) * self.W, axis=(1, 2))

    def gte(self, theta, a):
        y, ts = self.ate(theta, a), jnp.asarray(self.ts)
        return jnp.sum(0.5 * (y[1:] + y[:-1]) * jnp.diff(ts)) / ts[-1]

    def gradients(self, theta, a):
        """Values and Jacobians (last axis = parameters) of GTE, ATE(t) and tau at theta."""
        theta = jnp.asarray(theta)
        out = {}
        for name, f in (('GTE', self.gte), ('ATE', self.ate), ('tau', self.tau)):
            out[name] = np.asarray(f(theta, a))
            out['d' + name] = np.asarray(jax.jacfwd(f)(theta, a))
        return out

    # NumPy summaries of given fields (network output or FD)
    def summaries(self, C0f, CAf):
        tau = np.asarray(CAf) - np.asarray(C0f)
        ate = np.sum(tau * self.W, axis=(1, 2))
        trap = getattr(np, 'trapezoid', None) or np.trapz
        return {'tau': tau, 'ATE': ate, 'CE': trap(tau, self.ts, axis=0),
                'GTE': float(trap(ate, self.ts) / self.ts[-1]), 'peak_tau': float(tau.max()),
                'peak_ATE': float(ate.max()), 't_peak_ATE': float(self.ts[np.argmax(ate)])}


def scores(est, true):
    """Errors of an estimate's effect summaries against the truth (percent unless noted)."""
    rel = lambda e, r: float(100 * np.linalg.norm(e - r) / np.linalg.norm(r))
    return {'tau_L2_pct': rel(est['tau'], true['tau']),
            'tau_Linf': float(np.max(np.abs(est['tau'] - true['tau']))),
            'ATE_L2_pct': rel(est['ATE'], true['ATE']),
            'CE_L2_pct': rel(est['CE'], true['CE']),
            'GTE': est['GTE'], 'GTE_err_pct': 100 * (est['GTE'] - true['GTE']) / true['GTE'],
            'peak_tau': est['peak_tau'], 'peak_ATE': est['peak_ATE']}


def counterfactual_bound(F, grads, params):
    """Cramer-Rao bounds of the parameters and of the held-out effect (absolute units)."""
    Fi, cond = inverse(F)
    G = grads['dtau'].reshape(-1, len(params))
    return {'fisher_condition': cond,
            'sd_log_params': dict(zip(params, np.sqrt(np.clip(np.diag(Fi), 0, None)).tolist())),
            'sd_GTE': float(delta_sd(Fi, grads['dGTE'])),
            'sd_ATE': delta_sd(Fi, grads['dATE']),
            'rmse_L2_tau': float(np.sqrt(max(np.trace(Fi @ (G.T @ G)), 0.0)))}
