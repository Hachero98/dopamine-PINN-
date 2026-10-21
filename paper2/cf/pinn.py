"""Dose-aware PINN and the staged curriculum, for any mechanism.

Network C_theta(x, y, t, a); residual  dC/dt - D lap C + uptake(C; params, a).
Curriculum (paper 2, Section 5):
    stage 2  baseline parameters + network from untreated data (drug parameter pinned)
    stage 3  drug parameter + network from all training doses (baseline pinned at stage 2)
    stage 4  everything jointly
Collocation doses are 0 in stage 2 and uniform on [0, 1] afterwards, so the residual
also constrains the held-out dose, where there are no data.
Training is paper 1's trainer (Adam, then L-BFGS, fixed weights).
"""
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from dopamine_pinn.pinn import MLP, sample_points, train, TrainConfig
from .model import untreated


class DoseNet:
    """C(x, y, t, a) with raw inputs, as paper 1's network plus a dose input."""

    def __init__(self, layers):
        assert layers[0] == 4
        self.layers = tuple(layers)
        self.gdef = nnx.split(MLP(self.layers, rngs=nnx.Rngs(0)))[0]

    def init(self, seed):
        return nnx.split(MLP(self.layers, rngs=nnx.Rngs(seed)))[1]

    def fn(self, p_net):
        net = nnx.merge(self.gdef, p_net)
        return lambda z: net(z)[0]

    def predict(self, p_net, x, y, t, a, chunk=50_000):
        """Network values at arrays of points (broadcast together)."""
        x, y, t, a = np.broadcast_arrays(x, y, t, a)
        f = jax.jit(lambda pn, z: jax.vmap(self.fn(pn))(z))
        z = np.stack([np.ravel(v) for v in (x, y, t, a)], axis=1)
        out = np.concatenate([np.asarray(f(p_net, jnp.asarray(z[i:i + chunk]))) for i in range(0, len(z), chunk)])
        return out.reshape(x.shape)


_EX, _EY = jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0, 0.0])


def _derivs(f, z):
    """C, gradient (C_x, C_y, C_t, C_a) and Laplacian at one point z = (x, y, t, a)."""
    g = jax.grad(f)
    gz, hx = jax.jvp(g, (z,), (_EX,))
    _, hy = jax.jvp(g, (z,), (_EY,))
    return f(z), gz, hx[0] + hy[1]


def build_loss(mech, net, pin=None):
    """comps(p, batch) -> [L_res, L_init, L_wall, L_data]; p holds 'net' and 'log<param>'."""
    full = (lambda p: p) if not pin else (lambda p: {**p, **pin})

    def comps(p, b):
        q_ = full(p)
        f = net.fn(q_['net'])
        q = {k: jnp.exp(q_['log' + k]) for k in mech.params}

        def res(x, y, t, a):
            C, g, lap = _derivs(f, jnp.stack([x, y, t, a]))
            return g[2] - q['D'] * lap + mech.uptake(q, C, a)
        r = jax.vmap(res)(b['x_r'], b['y_r'], b['t_r'], b['a_r'])
        C_ic = jax.vmap(lambda x, y, a: f(jnp.stack([x, y, 0.0, a])))(b['x_i'], b['y_i'], b['a_i'])
        dn = jax.vmap(lambda x, y, t, a, nx, ny: jnp.dot(jax.grad(f)(jnp.stack([x, y, t, a]))[:2],
                                                         jnp.stack([nx, ny])))(
            b['x_b'], b['y_b'], b['t_b'], b['a_b'], b['nx_b'], b['ny_b'])
        C_d = jax.vmap(lambda x, y, t, a: f(jnp.stack([x, y, t, a])))(b['x_d'], b['y_d'], b['t_d'], b['a_d'])
        return jnp.stack([jnp.mean(r**2), jnp.mean((C_ic - b['C_i'])**2),
                          jnp.mean(dn**2), jnp.mean((C_d - b['C_d'])**2)])

    def no_rad(*_):
        raise NotImplementedError('adaptive resampling is not set up for the dose-aware network')

    comps.full, comps.net, comps.resid = full, net, no_rad
    return comps


def collocation(profile, design, seed):
    return sample_points(n_domain=profile.n_domain, n_initial=profile.n_initial, n_boundary=profile.n_boundary,
                         seed=seed, T_end=design.T, amp=design.C0)


def batch(pts, obs, dose_mode, seed):
    """Collocation + data as JAX arrays; dose_mode 'zero' or 'uniform' for collocation doses."""
    rng = np.random.default_rng(seed)
    b = dict(pts)
    for k, n in (('a_r', len(pts['x_r'])), ('a_i', len(pts['x_i'])), ('a_b', len(pts['x_b']))):
        b[k] = np.zeros(n) if dose_mode == 'zero' else rng.uniform(0.0, 1.0, n)
    b.update({k: obs[k] for k in ('x_d', 'y_d', 't_d', 'a_d', 'C_d')})
    b['t_scale'] = 1.0
    return {k: jnp.asarray(np.asarray(v, np.float64)) for k, v in b.items()}


def train_config(profile, verbose=True):
    return TrainConfig(adam_iters=profile.adam_iters, lbfgs_iters=profile.lbfgs_iters, lr=profile.lr,
                       n_chunk=profile.n_chunk, weights=profile.weights, verbose=verbose,
                       log_every=max(1, profile.adam_iters // profile.n_chunk // 4),
                       track_every=max(profile.n_chunk, profile.adam_iters // 40 // profile.n_chunk * profile.n_chunk))


def staged(mech, profile, design, obs, net_seed, pts_seed=None, verbose=True):
    """Stages 2-4. Returns (history, final p, net). history[stage] = params, losses, minutes, trajectory."""
    net = DoseNet(profile.layers)
    pts = collocation(profile, design, net_seed if pts_seed is None else pts_seed)
    b0 = batch(pts, untreated(obs), 'zero', net_seed)
    b1 = batch(pts, obs, 'uniform', net_seed + 1)
    tc = train_config(profile, verbose)
    lg = {f'log{k}': jnp.log(jnp.float64(v)) for k, v in mech.init.items()}
    dp = 'log' + mech.dose_param
    base = ['log' + k for k in mech.baseline_params]
    hist = {}

    def values(comps, p):
        return {k[3:]: float(jnp.exp(v)) for k, v in comps.full(p).items() if k.startswith('log')}

    def run(stage, p0, comps, b):
        report = lambda q: ' '.join(f'{k} {v:.5g}' for k, v in values(comps, q).items())
        p, info = train(p0, comps, b, tc, label=stage, report=report)
        hist[stage] = {'params': values(comps, p), 'L': info['L'], 'minutes': info['minutes'],
                       'trajectory': info['trajectory']}
        return p

    p2 = run('stage2', {'net': net.init(net_seed), **{k: lg[k] for k in base}},
             build_loss(mech, net, pin={dp: lg[dp]}), b0)
    p3 = run('stage3', {'net': p2['net'], dp: lg[dp]}, build_loss(mech, net, pin={k: p2[k] for k in base}), b1)
    p4 = run('stage4', {'net': p3['net'], **{k: p2[k] for k in base}, dp: p3[dp]}, build_loss(mech, net), b1)
    return hist, p4, net
