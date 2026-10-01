"""Physics-informed neural network: network, PDE residuals, collocation, training.

Parameters are a plain dict pytree: {'net': <Flax NNX state>, 'logD': ..., 'logk': ...}
(or 'logVmax', 'logKm' for Michaelis-Menten uptake). Physical parameters are learned in
log space. A loss is assembled from a `physics(p, x, y, t, C, grad, lap)` function that
returns the PDE residual at one point, so a new uptake law or a spatially varying D is a
new physics function, not a new trainer.

The trainer is the paper's schedule (Adam, then L-BFGS) with fixed weights by default;
learning-rate annealing of the weights (Wang, Teng & Perdikaris 2021) and residual-based
adaptive resampling (RAD; Wu et al. 2023) are optional.
"""
import time
from dataclasses import dataclass, field, replace

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree
from flax import nnx
import optax
from jaxopt import LBFGS
from scipy.stats import qmc

from . import config as cfg


# ---- network ----------------------------------------------------------------------
class MLP(nnx.Module):
    """tanh MLP with float64 parameters (a dtype-homogeneous pytree for jaxopt L-BFGS)."""

    def __init__(self, layers, *, rngs, zero_last=False):
        self.n_layers = len(layers) - 1
        for i in range(self.n_layers):
            last = zero_last and i == self.n_layers - 1
            setattr(self, f'lin_{i}', nnx.Linear(
                layers[i], layers[i + 1],
                kernel_init=(nnx.initializers.zeros_init() if last else nnx.initializers.glorot_normal()),
                bias_init=nnx.initializers.zeros_init(), param_dtype=jnp.float64, rngs=rngs))

    def __call__(self, z):
        h = z
        for i in range(self.n_layers - 1):
            h = jnp.tanh(getattr(self, f'lin_{i}')(h))
        return getattr(self, f'lin_{self.n_layers - 1}')(h)


class Net:
    """Concentration network C(x, y, t) with raw inputs (as in the paper).

    ff_features > 0 appends Fourier features sin/cos(2 pi B z) of the coordinates scaled
    to [0, 1] (z = (x / L, y / L, t / T_end)), B ~ N(0, ff_sigma^2) fixed by ff_seed.
    t_scale (passed at call time) multiplies the raw time input, so a long record can be
    mapped onto the paper's input range [0, 50].
    """

    def __init__(self, layers=cfg.LAYERS, ff_features=0, ff_sigma=3.0, ff_seed=2024, T_end=cfg.T, L=cfg.L):
        self.layers = (3 + 2 * ff_features,) + tuple(layers[1:])
        self.ff = ff_features
        self.gdef = nnx.split(MLP(self.layers, rngs=nnx.Rngs(0)))[0]
        self.B = (jnp.asarray(np.random.default_rng(ff_seed).normal(0.0, ff_sigma, (ff_features, 3)))
                  if ff_features else None)
        self.nrm = jnp.array([L, L, T_end])

    def init(self, seed):
        return nnx.split(MLP(self.layers, rngs=nnx.Rngs(seed)))[1]

    def fn(self, p_net, t_scale=1.0):
        net = nnx.merge(self.gdef, p_net)
        sc = jnp.stack([1.0, 1.0, jnp.asarray(t_scale, jnp.float64)])
        if not self.ff:
            return lambda z: net(z * sc)[0]
        B, nrm = self.B, self.nrm

        def f(z):
            zz = 2 * jnp.pi * (B @ (z / nrm))
            return net(jnp.concatenate([z * sc, jnp.sin(zz), jnp.cos(zz)]))[0]
        return f


EX, EY = jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])


def point_derivs(f, x, y, t):
    """C, gradient (C_x, C_y, C_t) and Laplacian C_xx + C_yy at one point."""
    z = jnp.stack([x, y, t])
    g = jax.grad(f)
    gz, hx = jax.jvp(g, (z,), (EX,))
    _, hy = jax.jvp(g, (z,), (EY,))
    return f(z), gz, hx[0] + hy[1]


# ---- physics: residual dC/dt - div(D grad C) + uptake ------------------------------
def phys_linear(p, x, y, t, C, g, lap):
    return g[2] - jnp.exp(p['logD']) * lap + jnp.exp(p['logk']) * C


def phys_mm(p, x, y, t, C, g, lap):
    """Michaelis-Menten uptake; |C| keeps the denominator positive early in training."""
    Vmax, Km = jnp.exp(p['logVmax']), jnp.exp(p['logKm'])
    return g[2] - jnp.exp(p['logD']) * lap + Vmax * C / (Km + jnp.abs(C))


class DField:
    """Spatially varying D(x, y) = D_ref exp(d(x, y)) from a small network (last layer zero)."""

    def __init__(self, layers=(2, 32, 32, 1), D_ref=cfg.D_INIT):
        self.layers, self.D_ref = tuple(layers), D_ref
        self.gdef = nnx.split(MLP(self.layers, rngs=nnx.Rngs(0), zero_last=True))[0]

    def init(self, seed):
        return nnx.split(MLP(self.layers, rngs=nnx.Rngs(seed), zero_last=True))[1]

    def fn(self, p_dnet):
        dn = nnx.merge(self.gdef, p_dnet)
        return lambda xy: self.D_ref * jnp.exp(dn(xy)[0])

    def physics(self):
        """Residual with div(D grad C) = D lap C + grad D . grad C and linear uptake."""
        def phys(p, x, y, t, C, g, lap):
            Df = self.fn(p['dnet'])
            xy = jnp.stack([x, y])
            D, gD = Df(xy), jax.grad(Df)(xy)
            return g[2] - (D * lap + gD[0] * g[0] + gD[1] * g[1]) + jnp.exp(p['logk']) * C
        return phys


# ---- loss terms ---------------------------------------------------------------------
def build_comps(physics, net, pin=None):
    """comps(p, batch) -> [L_r, L_i, L_b, L_d] (unweighted mean squares).

    pin: dict of entries held fixed (e.g. {'logD': log(0.32)}), merged into p.
    comps.resid(p, x, y, t, t_scale) gives the residual at arbitrary points (RAD).
    """
    full = (lambda p: p) if pin is None else (lambda p: {**p, **pin})

    def resid(p, x, y, t, ts):
        q = full(p)
        f = net.fn(q['net'], ts)
        return jax.vmap(lambda a, b_, c: physics(q, a, b_, c, *point_derivs(f, a, b_, c)))(x, y, t)

    def comps(p, b):
        q = full(p)
        f = net.fn(q['net'], b['t_scale'])
        res = resid(p, b['x_r'], b['y_r'], b['t_r'], b['t_scale'])
        C_ic = jax.vmap(lambda x, y: f(jnp.stack([x, y, 0.0])))(b['x_i'], b['y_i'])
        dn = jax.vmap(lambda x, y, t, nx, ny: jnp.dot(jax.grad(f)(jnp.stack([x, y, t]))[:2], jnp.stack([nx, ny])))(
            b['x_b'], b['y_b'], b['t_b'], b['nx_b'], b['ny_b'])
        C_d = jax.vmap(lambda x, y, t: f(jnp.stack([x, y, t])))(b['x_d'], b['y_d'], b['t_d'])
        return jnp.stack([jnp.mean(res**2), jnp.mean((C_ic - b['C_i'])**2),
                          jnp.mean(dn**2), jnp.mean((C_d - b['C_d'])**2)])

    comps.resid, comps.full, comps.net = resid, full, net
    return comps


# ---- collocation --------------------------------------------------------------------
def sample_points(n_domain=cfg.N_DOMAIN, n_initial=cfg.N_INITIAL, n_boundary=cfg.N_BOUNDARY_INV, seed=cfg.SEED,
                  obstacles=None, n_obstacle_bc=0, T_end=cfg.T, amp=cfg.C0, t_power=1.0, L=cfg.L, sigma=cfg.SIGMA):
    """Latin-hypercube interior points, uniform initial and wall points (+ obstacle walls).

    Draw order follows the original notebooks, so the default call gives the paper's
    collocation set for a given seed. obstacles: [(xc, yc, R), ...]; interior and initial
    points inside them are rejected and n_obstacle_bc points are spread over the circles
    in proportion to circumference, with the radial unit normal. t_power > 1 puts more
    interior points at early times (t = T_end u^p).
    """
    rng = np.random.default_rng(seed)
    outside = (lambda x, y: np.ones_like(x, bool)) if not obstacles else (
        lambda x, y: np.all([(x - a)**2 + (y - b)**2 > R**2 for a, b, R in obstacles], axis=0))
    over = 1.0 if not obstacles else 1.6
    u = qmc.LatinHypercube(d=3, seed=seed).random(int(over * n_domain))
    x_r, y_r, t_r = -L / 2 + L * u[:, 0], -L / 2 + L * u[:, 1], T_end * u[:, 2]**t_power
    keep = outside(x_r, y_r)
    x_r, y_r, t_r = x_r[keep][:n_domain], y_r[keep][:n_domain], t_r[keep][:n_domain]

    if not obstacles:
        x_i, y_i = rng.uniform(-L / 2, L / 2, n_initial), rng.uniform(-L / 2, L / 2, n_initial)
    else:
        x_i, y_i = np.empty(0), np.empty(0)
        while x_i.size < n_initial:
            cx, cy = rng.uniform(-L / 2, L / 2, 2 * n_initial), rng.uniform(-L / 2, L / 2, 2 * n_initial)
            ok = outside(cx, cy)
            x_i, y_i = np.concatenate([x_i, cx[ok]]), np.concatenate([y_i, cy[ok]])
        x_i, y_i = x_i[:n_initial], y_i[:n_initial]
    C_i = amp * np.exp(-(x_i**2 + y_i**2) / (2 * sigma**2))

    pe = n_boundary // 4
    s = lambda: rng.uniform(-L / 2, L / 2, pe)
    tt = lambda n: rng.uniform(0, T_end, n)
    edges = [(np.full(pe, -L / 2), s(), tt(pe), -np.ones(pe), np.zeros(pe)),
             (np.full(pe, L / 2), s(), tt(pe), np.ones(pe), np.zeros(pe)),
             (s(), np.full(pe, -L / 2), tt(pe), np.zeros(pe), -np.ones(pe)),
             (s(), np.full(pe, L / 2), tt(pe), np.zeros(pe), np.ones(pe))]
    if obstacles and n_obstacle_bc:
        circ = np.array([R for _, _, R in obstacles])
        for (a, b_, R), n in zip(obstacles, np.round(n_obstacle_bc * circ / circ.sum()).astype(int)):
            th = rng.uniform(0, 2 * np.pi, n)
            edges.append((a + R * np.cos(th), b_ + R * np.sin(th), tt(n), np.cos(th), np.sin(th)))
    cat = lambda i: np.concatenate([e[i] for e in edges])
    return {'x_r': x_r, 'y_r': y_r, 't_r': t_r, 'x_i': x_i, 'y_i': y_i, 'C_i': C_i,
            'x_b': cat(0), 'y_b': cat(1), 't_b': cat(2), 'nx_b': cat(3), 'ny_b': cat(4)}


def make_batch(pts, obs=None, t_scale=1.0):
    """Collocation points + observations as float64 JAX arrays (obs None: one dummy point)."""
    b = dict(pts)
    if obs is None:
        obs = {'x_d': [0.0], 'y_d': [0.0], 't_d': [0.0], 'C_d': [0.0]}
    b.update({k: obs[k] for k in ('x_d', 'y_d', 't_d', 'C_d')})
    b['t_scale'] = t_scale
    return {k: jnp.asarray(np.asarray(v, np.float64)) for k, v in b.items()}


# ---- training -----------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    """Training schedule and optional remedies. Defaults = the paper's inverse setting."""
    adam_iters: int = cfg.ADAM_ITERS
    lbfgs_iters: int = cfg.LBFGS_ITERS
    lr: float = cfg.LR
    n_chunk: int = cfg.N_CHUNK
    weights: tuple = cfg.W_INVERSE
    anneal: bool = False          # learning-rate annealing of weights 1..3 (Wang et al. 2021)
    anneal_every: int = 100
    anneal_alpha: float = 0.9
    anneal_cap: float = 100.0     # weights kept within [w / cap, w * cap]; None = uncapped
    rad: bool = False             # residual-based adaptive resampling (Wu et al. 2023)
    rad_every: int = 2_000
    rad_pool: int = 4             # pool size = rad_pool x number of interior points
    rad_k: float = 1.0
    rad_c: float = 1.0
    log_every: int = 80           # print every log_every chunks
    track_every: int = 500        # record the physical parameters every track_every Adam steps
    verbose: bool = True

    @classmethod
    def quick(cls, **kw):
        """Tiny schedule for smoke tests (not for results)."""
        return cls(**{**dict(adam_iters=200, lbfgs_iters=20, n_chunk=50, anneal_every=100, rad_every=100,
                             log_every=1000, verbose=False), **kw})

    def replace(self, **kw):
        return replace(self, **kw)


def train(p, comps, batch, tc=TrainConfig(), label='', report=None, rad_pool=None):
    """Adam then L-BFGS on sum_j w_j L_j. Returns (params, info).

    rad_pool(step) -> (x, y, t) candidate interior points; used when tc.rad is True.
    info: final weights, weight history, parameter trajectory, final batch, loss terms, minutes.
    """
    assert tc.adam_iters % tc.n_chunk == 0
    w = jnp.asarray(tc.weights, jnp.float64)
    w0 = w
    total = lambda q, b, w_: jnp.dot(w_, comps(q, b))
    opt = optax.adam(tc.lr)
    state = opt.init(p)

    @jax.jit
    def chunk(q, s, b, w_):
        def body(_, carry):
            q, s, _l = carry
            l, g = jax.value_and_grad(total)(q, b, w_)
            u, s = opt.update(g, s, q)
            return optax.apply_updates(q, u), s, l
        return lax.fori_loop(0, tc.n_chunk, body, (q, s, jnp.zeros((), jnp.float64)))

    @jax.jit
    def lam_hat(q, b, w_):
        def gnet(j, scale):
            return jnp.abs(ravel_pytree(jax.grad(lambda qq: scale * comps(qq, b)[j])(q)['net'])[0])
        g_r = gnet(0, w_[0])
        return jnp.stack([jnp.max(g_r) / jnp.maximum(jnp.mean(gnet(j, 1.0)), 1e-30) for j in (1, 2, 3)])

    pool_resid = jax.jit(comps.resid)
    report = report or (lambda q: '')
    hist = [(0, [float(v) for v in w])]
    phys = lambda q: {k[3:]: float(jnp.exp(v)) for k, v in comps.full(q).items() if k.startswith('log')}
    traj = [(0, phys(p))]
    t0 = time.time()
    n_chunks = tc.adam_iters // tc.n_chunk
    for c in range(n_chunks):
        p, state, l = chunk(p, state, batch, w)
        step = (c + 1) * tc.n_chunk
        if step % tc.track_every == 0:
            traj.append((step, phys(p)))
        if tc.anneal and step % tc.anneal_every == 0:
            w = w.at[1:].set((1 - tc.anneal_alpha) * w[1:] + tc.anneal_alpha * lam_hat(p, batch, w))
            if tc.anneal_cap is not None:
                w = w.at[1:].set(jnp.clip(w[1:], w0[1:] / tc.anneal_cap, w0[1:] * tc.anneal_cap))
            if step % 1000 == 0:
                hist.append((step, [float(v) for v in w]))
        if tc.rad and rad_pool is not None and step % tc.rad_every == 0 and step < tc.adam_iters:
            x, y, t = rad_pool(step)
            r = np.concatenate([np.asarray(pool_resid(p, jnp.asarray(x[i:i + 20_000]), jnp.asarray(y[i:i + 20_000]),
                                                      jnp.asarray(t[i:i + 20_000]), batch['t_scale']))
                                for i in range(0, len(x), 20_000)])
            prob = np.abs(r)**tc.rad_k / np.mean(np.abs(r)**tc.rad_k) + tc.rad_c
            idx = np.random.default_rng(step).choice(len(x), batch['x_r'].shape[0], replace=False, p=prob / prob.sum())
            batch = {**batch, 'x_r': jnp.asarray(x[idx]), 'y_r': jnp.asarray(y[idx]), 't_r': jnp.asarray(t[idx])}
        if tc.verbose and (c == 0 or (c + 1) % tc.log_every == 0 or c == n_chunks - 1):
            print(f'    [{label}] step {step:>5d}  loss {float(l):.4e}  {report(p)}  {time.time() - t0:.0f}s')
    if tc.lbfgs_iters:
        p = LBFGS(fun=lambda q, b: total(q, b, w), maxiter=tc.lbfgs_iters, tol=1e-9).run(p, b=batch).params
    if tc.verbose:
        print(f'    [{label}] after L-BFGS: {report(p)}  total {time.time() - t0:.0f}s')
    hist.append((tc.adam_iters, [float(v) for v in w]))
    traj.append((tc.adam_iters + tc.lbfgs_iters, phys(p)))
    terms = np.asarray(comps(p, batch))
    return p, {'weights': [float(v) for v in w], 'weight_history': hist, 'trajectory': traj, 'batch': batch,
               'L': dict(zip(('r', 'i', 'b', 'd'), map(float, terms))),
               'loss': float(np.dot(np.asarray(w), terms)), 'minutes': (time.time() - t0) / 60}


def predict(net, p, x, y, t, t_scale=1.0, chunk=50_000):
    """Network concentrations at arrays of points (batched)."""
    f_batch = jax.jit(lambda pn, xyt: jax.vmap(net.fn(pn, t_scale))(xyt))
    xyt = np.stack([np.ravel(x), np.ravel(y), np.ravel(t)], axis=1)
    out = np.concatenate([np.asarray(f_batch(p['net'], jnp.asarray(xyt[i:i + chunk])))
                          for i in range(0, len(xyt), chunk)])
    return out.reshape(np.shape(x))
