"""Run settings: the experimental design and the training profile.

Design  what is observed and predicted (doses, held-out dose, points, noise, release).
Profile how hard the PINN trains (collocation points, iterations, loss weights).

Profiles
    smoke  seconds on a CPU; for tests only, numbers mean nothing
    fast   a few minutes per study on a Colab GPU; for checking that things work and
           for quick comparisons (e.g. loss weights)
    paper  paper 1's settings (30k collocation points, 20k Adam + 2k L-BFGS per stage)

Change a setting with .replace(...), e.g. PROFILES['fast'].replace(w_res=30_000).
"""
from dataclasses import dataclass, replace, asdict

from dopamine_pinn import config as cfg


@dataclass(frozen=True)
class Design:
    doses: tuple = (0.0, 0.25, 0.75, 1.0)   # training doses (observed)
    heldout: float = 0.5                     # predicted, never observed
    n_obs: int = cfg.N_OBS                   # scattered points per dose
    noise_sd: float = cfg.NOISE_SD           # uM
    C0: float = cfg.C0                       # peak release, uM
    T: float = cfg.T                         # record length, ms
    dose_seed_step: int = 100                # dose j uses obs_seed + 100 j

    def __post_init__(self):
        assert self.heldout not in self.doses, 'the held-out dose must never be observed'
        assert 0.0 in self.doses, 'stage 2 needs untreated data (dose 0)'

    @property
    def t_window(self):
        return (0.5, 0.9 * self.T)

    def replace(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Profile:
    name: str
    n_domain: int            # interior collocation points
    n_initial: int
    n_boundary: int
    adam_iters: int          # per stage
    lbfgs_iters: int         # per stage
    layers: tuple = (4, 64, 64, 64, 64, 1)
    w_res: float = cfg.W_INVERSE[0]
    w_init: float = cfg.W_INVERSE[1]
    w_bound: float = cfg.W_INVERSE[2]
    w_data: float = cfg.W_INVERSE[3]
    lr: float = cfg.LR
    n_chunk: int = 50
    eval_dt: float = 1.0     # ms between evaluation frames (effects)
    fd_nx: int = cfg.NX_FD   # FD nodes per side (data, reference, bound, truth)

    @property
    def weights(self):
        return (self.w_res, self.w_init, self.w_bound, self.w_data)

    def replace(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)


PROFILES = {
    'smoke': Profile('smoke', n_domain=300, n_initial=100, n_boundary=100, adam_iters=200, lbfgs_iters=20,
                     layers=(4, 16, 16, 1), eval_dt=5.0, fd_nx=31),
    'fast': Profile('fast', n_domain=3_000, n_initial=400, n_boundary=400, adam_iters=5_000, lbfgs_iters=500),
    'paper': Profile('paper', n_domain=cfg.N_DOMAIN, n_initial=cfg.N_INITIAL, n_boundary=cfg.N_BOUNDARY_INV,
                     adam_iters=cfg.ADAM_ITERS, lbfgs_iters=cfg.LBFGS_ITERS),
}
