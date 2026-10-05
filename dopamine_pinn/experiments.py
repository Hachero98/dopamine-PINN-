"""High-level runners for the experiments of the baseline paper.

Each function returns a plain dict of numbers (JSON-serialisable), so notebooks and
scripts only orchestrate and save. Later studies reuse `inverse` with another physics
function, observation design or TrainConfig.
"""
import numpy as np
import jax.numpy as jnp

from . import config as cfg
from .solver import make_fd_solver, solve_at, linear_sink, interp_frames, analytical_infinite
from .pinn import Net, build_comps, sample_points, make_batch, train, predict, TrainConfig, phys_linear, phys_mm
from .reference import fit_linear


def pct(est, true):
    return float(100.0 * (est - true) / true)


def truth_frames(D=cfg.D_TRUE, k=cfg.K_TRUE, solver=None):
    solver = solver or make_fd_solver()
    return solver, solve_at(solver, D, linear_sink(k))


def _report(comps):
    return lambda q: ' '.join(f'{k[3:]} {float(jnp.exp(v)):.5f}' for k, v in comps.full(q).items()
                              if k.startswith('log'))


def inverse(obs, tc=TrainConfig(), seed=cfg.SEED, physics='linear', init=None, net=None,
            pts=None, pts_seed=None, t_scale=1.0, pin=None, label='inverse', rad_pool=None, **pts_kw):
    """Fit the PINN to observations. physics: 'linear', 'mm' or a physics function.

    init: dict of initial log-parameters (defaults: D_INIT, K_INIT or the MM offsets).
    pts: collocation set; otherwise drawn with sample_points(seed=pts_seed or seed, **pts_kw).
    Returns recovered parameters, loss terms, weights and minutes.
    """
    phys = {'linear': phys_linear, 'mm': phys_mm}.get(physics, physics)
    if init is None:
        init = ({'logD': np.log(cfg.D_INIT), 'logk': np.log(cfg.K_INIT)} if physics == 'linear' else
                {'logD': np.log(cfg.D_INIT), 'logVmax': np.log(0.8 * cfg.VMAX_TRUE),
                 'logKm': np.log(1.2 * cfg.KM_TRUE)})
    net = net or Net()
    comps = build_comps(phys, net, pin=pin)
    if pts is None:
        pts = sample_points(seed=seed if pts_seed is None else pts_seed, **pts_kw)
    batch = make_batch(pts, obs, t_scale)
    p0 = {'net': net.init(seed), **{k: jnp.asarray(v) for k, v in init.items() if not (pin and k in pin)}}
    p, info = train(p0, comps, batch, tc, label=label, report=_report(comps), rad_pool=rad_pool)
    q = comps.full(p)
    return {'params': {k[3:]: float(jnp.exp(v)) for k, v in q.items() if k.startswith('log')},
            'L': info['L'], 'loss': info['loss'], 'weights': info['weights'],
            'rms_data': float(np.sqrt(info['L']['d'])), 'minutes': info['minutes'],
            'trajectory': info['trajectory'], '_p': p, '_net': net, '_comps': comps, '_batch': info['batch']}


def linear_errors(res, ref=None, D_true=cfg.D_TRUE, k_true=cfg.K_TRUE):
    out = {'D': res['params']['D'], 'k': res['params']['k'],
           'D_err': pct(res['params']['D'], D_true), 'k_err': pct(res['params']['k'], k_true)}
    if ref is not None:
        out.update({'ref_D': ref['D'], 'ref_k': ref['k'], 'ref_D_err': pct(ref['D'], D_true),
                    'ref_k_err': pct(ref['k'], k_true), 'ref_sd_logD': ref['sd_logD'],
                    'ref_sd_logk': ref['sd_logk'], 'ref_rho': ref['rho']})
    return out


def forward(tc=None, seed=cfg.SEED, layers=cfg.LAYERS, D=cfg.D_TRUE, k=cfg.K_TRUE):
    """Forward PINN at known (D, k) with weights (10, 1, 1, 0) and 400 wall points.

    Errors are relative L2 against the FD solution on the paper's 41 x 41 x 51 grid
    (whole window and per snapshot), and against the infinite-domain analytical solution.
    """
    tc = tc or TrainConfig(weights=cfg.W_FORWARD)
    net = Net(layers=layers)
    pin = {'logD': jnp.log(D), 'logk': jnp.log(k)}
    comps = build_comps(phys_linear, net, pin=pin)
    batch = make_batch(sample_points(n_boundary=cfg.N_BOUNDARY_FWD, seed=seed), None)
    p, info = train({'net': net.init(seed)}, comps, batch, tc, label=f'forward {layers}')
    solver, fr = truth_frames(D, k)
    xs = np.linspace(-cfg.L / 2, cfg.L / 2, 41)
    ts = np.linspace(0, cfg.T, int(cfg.T) + 1)
    X, Y, Tg = np.meshgrid(xs, xs, ts, indexing='ij')
    C_pinn = predict(net, p, X, Y, Tg)
    C_fd = np.asarray(interp_frames(fr, solver.dx, jnp.asarray(X.ravel()), jnp.asarray(Y.ravel()),
                                    jnp.asarray(Tg.ravel()))).reshape(X.shape)
    C_an = analytical_infinite(X, Y, Tg, D, k)
    rel = lambda a, b: float(100 * np.linalg.norm(a - b) / np.linalg.norm(b))
    snaps = {}
    for t_s in (1.0, 5.0, 10.0, 20.0, 50.0):
        i = int(np.argmin(np.abs(ts - t_s)))
        snaps[f't={t_s:g}'] = {'vs_fd': rel(C_pinn[..., i], C_fd[..., i]),
                               'vs_analytical': rel(C_pinn[..., i], C_an[..., i])}
    n_params = sum(net.layers[i] * net.layers[i + 1] + net.layers[i + 1] for i in range(len(net.layers) - 1))
    return {'layers': list(net.layers), 'n_params': n_params, 'L2_vs_fd': rel(C_pinn, C_fd),
            'L2_vs_analytical': rel(C_pinn, C_an), 'snapshots': snaps, 'minutes': info['minutes'],
            '_p': p, '_net': net}


def inverse_with_reference(obs, solver, tc=TrainConfig(), seed=cfg.SEED, noise_sd=cfg.NOISE_SD,
                           D_true=cfg.D_TRUE, k_true=cfg.K_TRUE, label='inverse', **kw):
    """PINN and reference estimator on the same data; signed errors of both."""
    ref = fit_linear(obs, solver, noise_sd=max(noise_sd, 1e-6))
    res = inverse(obs, tc, seed=seed, label=label, **kw)
    out = linear_errors(res, ref, D_true, k_true)
    out.update({'minutes': res['minutes'], 'L': res['L'], 'rms_data_over_sigma':
                res['rms_data'] / noise_sd if noise_sd > 0 else None})
    return out, res
