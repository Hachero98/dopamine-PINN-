"""Forward model (FD solver) for any mechanism, and the synthetic observations.

Parameters travel as theta = log values in the order of mechanism.params. The FD solver
is differentiable (JAX), so the reference fit, the Fisher information and the effect
gradients all come from the same code.
"""
import numpy as np
import jax.numpy as jnp

from dopamine_pinn import config as cfg
from dopamine_pinn.solver import make_fd_solver, gaussian_ic, interp_frames
from dopamine_pinn.observations import scattered, merge, subset


class Forward:
    def __init__(self, mech, design, nx=cfg.NX_FD):
        self.mech, self.design = mech, design
        self.solver = make_fd_solver(nx=nx, T_end=design.T)
        self.C_init = jnp.asarray(gaussian_ic(nx, design.C0, L=cfg.L))
        self.theta_true = self.theta(mech.truth)
        self.theta_init = self.theta(mech.init)

    # parameter conversions
    def theta(self, values):
        return jnp.log(jnp.asarray([values[p] for p in self.mech.params], jnp.float64))

    def values(self, theta):
        return {p: float(np.exp(theta[i])) for i, p in enumerate(self.mech.params)}

    def q(self, theta):
        return {p: jnp.exp(theta[i]) for i, p in enumerate(self.mech.params)}

    # solutions
    def frames(self, theta, a):
        """FD frames of C^(a), every dt_store ms on [0, T]."""
        q = self.q(theta)
        return self.solver(q['D'], lambda C: self.mech.uptake(q, C, a), self.C_init)

    def predictor(self, obs, free=None, fixed=None):
        """theta_free -> C at the observations (JAX). Parameters not in `free` are taken
        from `fixed` (natural values)."""
        params = self.mech.params
        free = list(params if free is None else free)
        fixed = fixed or {}
        x, y, t, a = (np.asarray(obs[k]) for k in ('x_d', 'y_d', 't_d', 'a_d'))
        groups = [(float(d), np.flatnonzero(a == d)) for d in np.unique(a)]
        s = self.solver

        def predict(th_free):
            vals = {p: jnp.log(jnp.float64(v)) for p, v in fixed.items()}
            vals.update({p: th_free[i] for i, p in enumerate(free)})
            theta = jnp.stack([vals[p] for p in params])
            out = jnp.zeros(len(x))
            for d, idx in groups:
                fr = self.frames(theta, d)
                out = out.at[idx].set(interp_frames(fr, s.dx, jnp.asarray(x[idx]), jnp.asarray(y[idx]),
                                                    jnp.asarray(t[idx]), s.dt_store, s.L))
            return out
        return predict


def observations(fwd, seed, theta=None):
    """Scattered points at every training dose (the draws of paper 1 and the causal runs)."""
    d = fwd.design
    theta = fwd.theta_true if theta is None else theta
    sets = []
    for j, a in enumerate(d.doses):
        o = scattered(fwd.frames(theta, a), fwd.solver, seed + d.dose_seed_step * j,
                      n_obs=d.n_obs, noise_sd=d.noise_sd, t_window=d.t_window)
        o['a_d'] = np.full(d.n_obs, float(a))
        sets.append(o)
    return merge(*sets)


def untreated(obs):
    return subset(obs, np.asarray(obs['a_d']) == 0.0)
