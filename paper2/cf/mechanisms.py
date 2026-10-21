"""Uptake mechanisms and how the drug acts on them.

Every model is  dC/dt = D lap C - uptake(C; params, a)  with zero-flux walls and a
Gaussian release; the drug dose a in [0, 1] enters only through the uptake. A mechanism
lists its parameters, their true and starting values, which one carries the drug effect
(`dose_param`, not identifiable from untreated data and pinned in stage 2) and the
uptake law. All parameters are positive and are fitted on the log scale.

To add a model: subclass Mechanism, set the class attributes, write `uptake`, and add it
to MECHANISMS. Nothing else in the pipeline changes.
"""
import jax.numpy as jnp

from dopamine_pinn import config as cfg


class Mechanism:
    name = ''
    params = ()          # all physical parameters, D first
    dose_param = ''      # the drug-sensitivity parameter
    TRUTH = {}
    INIT = {}            # offset starting values (as in paper 1)
    BOUNDS = {}          # (low, high) for the reference fit; None = no bound
    description = ''

    def __init__(self, truth=None, init=None):
        self.truth = {**self.TRUTH, **(truth or {})}
        self.init = {**self.INIT, **(init or {})}
        assert set(self.truth) == set(self.params) == set(self.init)
        assert self.dose_param in self.params and self.params[0] == 'D'

    @property
    def baseline_params(self):
        return tuple(p for p in self.params if p != self.dose_param)

    def uptake(self, q, C, a):
        """Uptake rate (uM/ms); q maps parameter names to values (natural units)."""
        raise NotImplementedError

    def to_dict(self):
        return {'name': self.name, 'description': self.description, 'params': list(self.params),
                'dose_param': self.dose_param, 'truth': self.truth, 'init': self.init}


class Linear(Mechanism):
    name = 'linear'
    params = ('D', 'k', 'eta')
    dose_param = 'eta'
    TRUTH = {'D': cfg.D_TRUE, 'k': cfg.K_TRUE, 'eta': 0.6}
    INIT = {'D': cfg.D_INIT, 'k': cfg.K_INIT, 'eta': 0.4}
    BOUNDS = {'D': (0.02, None), 'k': (1e-4, 0.5), 'eta': (1e-3, 0.999)}
    description = 'k(a) = k (1 - eta a): the drug removes a fraction eta a of the uptake rate'

    def uptake(self, q, C, a):
        return q['k'] * (1.0 - q['eta'] * a) * C


class MMCapacity(Mechanism):
    """Michaelis-Menten; the drug removes transporters (non-competitive block, DAT loss)."""
    name = 'mm_capacity'
    params = ('D', 'Vmax', 'Km', 'eta')
    dose_param = 'eta'
    TRUTH = {'D': cfg.D_TRUE, 'Vmax': cfg.VMAX_TRUE, 'Km': cfg.KM_TRUE, 'eta': 0.6}
    INIT = {'D': cfg.D_INIT, 'Vmax': 0.8 * cfg.VMAX_TRUE, 'Km': 1.2 * cfg.KM_TRUE, 'eta': 0.4}
    BOUNDS = {'D': (0.02, None), 'Vmax': (1e-5, 0.5), 'Km': (1e-3, 50.0), 'eta': (1e-3, 0.999)}
    description = 'Vmax(a) = Vmax (1 - eta a): capacity loss'

    def uptake(self, q, C, a):
        return q['Vmax'] * (1.0 - q['eta'] * a) * C / (q['Km'] + jnp.abs(C))


class MMCompetitive(Mechanism):
    """Michaelis-Menten; the drug competes with dopamine (cocaine/methylphenidate-like).

    gamma = 1.5 makes the low-concentration rate at a = 1 equal to the linear model's,
    Vmax / (Km (1 + gamma)) = 0.4 Vmax / Km, so the three models match where C << Km.
    """
    name = 'mm_competitive'
    params = ('D', 'Vmax', 'Km', 'gamma')
    dose_param = 'gamma'
    TRUTH = {'D': cfg.D_TRUE, 'Vmax': cfg.VMAX_TRUE, 'Km': cfg.KM_TRUE, 'gamma': 1.5}
    INIT = {'D': cfg.D_INIT, 'Vmax': 0.8 * cfg.VMAX_TRUE, 'Km': 1.2 * cfg.KM_TRUE, 'gamma': 1.0}
    BOUNDS = {'D': (0.02, None), 'Vmax': (1e-5, 0.5), 'Km': (1e-3, 50.0), 'gamma': (1e-3, 100.0)}
    description = 'Km(a) = Km (1 + gamma a): competitive block'

    def uptake(self, q, C, a):
        return q['Vmax'] * C / (q['Km'] * (1.0 + q['gamma'] * a) + jnp.abs(C))


MECHANISMS = {m.name: m for m in (Linear, MMCapacity, MMCompetitive)}


def get(name, **kw):
    return MECHANISMS[name](**kw)
