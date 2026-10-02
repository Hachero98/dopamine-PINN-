"""Mechanistic counterfactual analysis, do(A = a), on the linear model.

The causal-inference layer of the Level 2 research plan (Level2_Dopamine_PINN_v1.pdf,
Sections 0, 3, 5, 6) on the baseline linear model, before the nonlinear extension.

Intervention. A in [0, 1] is the fractional DAT occupancy of a reuptake inhibitor,
applied as do(A = a): the uptake mechanism is set, not conditioned on. Assumed pathway

    A  ->  k(A) = k (1 - eta A)  ->  C(x, y, t),

so the drug changes only the linearised uptake rate; D, the release, the geometry and
the walls do not depend on A. k is the baseline (untreated) rate of the paper, eta the
treatment sensitivity. The PDE is the paper's, dC/dt = D lap C - k (1 - eta A) C.

Quantities. C^(a)(x, y, t) is the potential-outcome field under do(A = a),
tau_a = C^(a) - C^(0) the pointwise intervention effect, and with a dose-aware network
C_theta(x, y, t, a) the estimate tau_theta,a = C_theta^(a) - C_theta^(0). Summaries:
ATE_a(t) (domain average), CE_a(x, y) (time integral), GTE_a (space-time average),
peak effect and dose sensitivity S_A = dC^(a)/da. These are mechanistic counterfactual
effects, valid within the assumed PDE, not nonparametric causal estimates.

Identifiability. At A = 0 the model does not depend on eta, so untreated data carry no
information on it; with two or more doses k(A) is a line whose intercept (k) and slope
(-k eta) are both identifiable. `identifiability` checks this with the exact Jacobian.

Curriculum (Section 5), built on `pinn.train` with `pin` holding parameters fixed:
    stage 2  (D, k) and the network from A = 0 data only (eta pinned)
    stage 3  eta and the network from all training doses, (D, k) pinned at stage 2
    stage 4  everything jointly
    stage 5  held-out dose A = 0.5 (never observed): tau, ATE, CE, GTE, S_A
The reference estimator (least squares through the FD solver, Laplace covariance) is fit
on the same data in the same stages; its delta-method SDs give the information limit
of the design for every effect, which the PINN is compared with.

The A = 0 observations with the default seed are exactly the benchmark set
`linear_s4001`, so stage 2 is the paper's canonical inverse problem.

Moving to the nonlinear model means a new `physics` function and the matching `sink`.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from . import config as cfg
from .solver import make_fd_solver, solve_at, linear_sink, interp_frames, gaussian_ic, total_mass
from .observations import scattered, merge, subset
from .reference import ref_fit
from .pinn import MLP, sample_points, train, TrainConfig

# ---- treatment design ----------------------------------------------------------------
ETA_TRUE = 0.6                  # treatment sensitivity, dimensionless
ETA_INIT = 0.4                  # offset initial guess (-33%)
A_TRAIN = (0.0, 0.25, 0.75, 1.0)
A_HELDOUT = 0.5                 # never observed; stage 5 only
OBS_SEED = 4001                 # dose 0 gives the benchmark set linear_s4001
DOSE_SEED_STEP = 100            # dose j uses OBS_SEED + 100 j (independent noise per dose)
LAYERS = (4, 64, 64, 64, 64, 1)  # (x, y, t, a) -> C
PARAMS = ('D', 'k', 'eta')
TRUE = {'D': cfg.D_TRUE, 'k': cfg.K_TRUE, 'eta': ETA_TRUE}
INIT = {'D': cfg.D_INIT, 'k': cfg.K_INIT, 'eta': ETA_INIT}


def k_of(a, k=cfg.K_TRUE, eta=ETA_TRUE):
    return k * (1.0 - eta * a)


def truth_frames(a, solver, D=cfg.D_TRUE, k=cfg.K_TRUE, eta=ETA_TRUE):
    """FD frames of the potential outcome C^(a)."""
    return solve_at(solver, D, linear_sink(k_of(a, k, eta)))


# ---- observations ---------------------------------------------------------------------
def dose_observations(solver, doses=A_TRAIN, seed=OBS_SEED, n_obs=cfg.N_OBS, noise_sd=cfg.NOISE_SD,
                      D=cfg.D_TRUE, k=cfg.K_TRUE, eta=ETA_TRUE):
    """Scattered design (the paper's) at every training dose, tagged with a_d.

    Dose j draws locations, times and noise from seed + DOSE_SEED_STEP * j.
    """
    assert A_HELDOUT not in doses, 'the held-out dose must never be observed'
    sets = []
    for j, a in enumerate(doses):
        o = scattered(truth_frames(a, solver, D, k, eta), solver, seed + DOSE_SEED_STEP * j,
                      n_obs=n_obs, noise_sd=noise_sd)
        o['a_d'] = np.full(n_obs, float(a))
        sets.append(o)
    return merge(*sets)


def untreated(obs):
    return subset(obs, np.asarray(obs['a_d']) == 0.0)


# ---- dose-aware network and residual --------------------------------------------------
class DoseNet:
    """Concentration network C(x, y, t, a), raw inputs as in the paper's Net."""

    def __init__(self, layers=LAYERS):
        self.layers = tuple(layers)
        self.gdef = nnx.split(MLP(self.layers, rngs=nnx.Rngs(0)))[0]

    def init(self, seed):
        return nnx.split(MLP(self.layers, rngs=nnx.Rngs(seed)))[1]

    def fn(self, p_net, t_scale=1.0):
        net = nnx.merge(self.gdef, p_net)
        sc = jnp.stack([1.0, 1.0, jnp.asarray(t_scale, jnp.float64), 1.0])
        return lambda z: net(z * sc)[0]


EX4, EY4 = jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0, 0.0])


def point_derivs(f, x, y, t, a):
    """C, gradient (C_x, C_y, C_t, C_a) and Laplacian C_xx + C_yy at one point."""
    z = jnp.stack([x, y, t, a])
    g = jax.grad(f)
    gz, hx = jax.jvp(g, (z,), (EX4,))
    _, hy = jax.jvp(g, (z,), (EY4,))
    return f(z), gz, hx[0] + hy[1]


def phys_dose_linear(p, x, y, t, a, C, g, lap):
    """dC/dt - D lap C + k (1 - eta a) C."""
    k_a = jnp.exp(p['logk']) * (1.0 - jnp.exp(p['logeta']) * a)
    return g[2] - jnp.exp(p['logD']) * lap + k_a * C


def build_dose_comps(physics, net, pin=None):
    """comps(p, batch) -> [L_r, L_i, L_b, L_d], the dose-aware twin of pinn.build_comps."""
    full = (lambda p: p) if pin is None else (lambda p: {**p, **pin})

    def resid(p, x, y, t, a, ts):
        q = full(p)
        f = net.fn(q['net'], ts)
        return jax.vmap(lambda x_, y_, t_, a_: physics(q, x_, y_, t_, a_, *point_derivs(f, x_, y_, t_, a_)))(
            x, y, t, a)

    def comps(p, b):
        q = full(p)
        f = net.fn(q['net'], b['t_scale'])
        res = resid(p, b['x_r'], b['y_r'], b['t_r'], b['a_r'], b['t_scale'])
        C_ic = jax.vmap(lambda x, y, a: f(jnp.stack([x, y, 0.0, a])))(b['x_i'], b['y_i'], b['a_i'])
        dn = jax.vmap(lambda x, y, t, a, nx, ny: jnp.dot(jax.grad(f)(jnp.stack([x, y, t, a]))[:2],
                                                         jnp.stack([nx, ny])))(
            b['x_b'], b['y_b'], b['t_b'], b['a_b'], b['nx_b'], b['ny_b'])
        C_d = jax.vmap(lambda x, y, t, a: f(jnp.stack([x, y, t, a])))(b['x_d'], b['y_d'], b['t_d'], b['a_d'])
        return jnp.stack([jnp.mean(res**2), jnp.mean((C_ic - b['C_i'])**2),
                          jnp.mean(dn**2), jnp.mean((C_d - b['C_d'])**2)])

    def no_rad(*_):
        raise NotImplementedError('adaptive resampling is not set up for the dose-aware network')

    comps.resid, comps.full, comps.net = no_rad, full, net
    return comps


def collocation_doses(pts, mode, seed):
    """Dose coordinate of every collocation point: 'zero' (stage 2) or U[0, 1] (stages 3-4).

    U[0, 1] lets the residual constrain the held-out slice a = 0.5 without data there.
    """
    rng = np.random.default_rng(seed)
    n = {'a_r': len(pts['x_r']), 'a_i': len(pts['x_i']), 'a_b': len(pts['x_b'])}
    if mode == 'zero':
        return {k: np.zeros(v) for k, v in n.items()}
    return {k: rng.uniform(0.0, 1.0, v) for k, v in n.items()}


def make_dose_batch(pts, a_pts, obs, t_scale=1.0):
    b = {**pts, **a_pts, 't_scale': t_scale}
    b.update({k: obs[k] for k in ('x_d', 'y_d', 't_d', 'a_d', 'C_d')})
    return {k: jnp.asarray(np.asarray(v, np.float64)) for k, v in b.items()}


def predict_dose(net, p, x, y, t, a, t_scale=1.0, chunk=50_000):
    """Network concentrations at arrays of points (x, y, t, a broadcast together)."""
    x, y, t, a = np.broadcast_arrays(x, y, t, a)
    f_batch = jax.jit(lambda pn, z: jax.vmap(net.fn(pn, t_scale))(z))
    z = np.stack([np.ravel(v) for v in (x, y, t, a)], axis=1)
    out = np.concatenate([np.asarray(f_batch(p['net'], jnp.asarray(z[i:i + chunk])))
                          for i in range(0, len(z), chunk)])
    return out.reshape(x.shape)


def dose_sensitivity(net, p, x, y, t, a, t_scale=1.0, chunk=50_000):
    """S_A = dC_theta/da by automatic differentiation."""
    x, y, t, a = np.broadcast_arrays(x, y, t, a)
    g_batch = jax.jit(lambda pn, z: jax.vmap(lambda zz: jax.grad(net.fn(pn, t_scale))(zz)[3])(z))
    z = np.stack([np.ravel(v) for v in (x, y, t, a)], axis=1)
    out = np.concatenate([np.asarray(g_batch(p['net'], jnp.asarray(z[i:i + chunk])))
                          for i in range(0, len(z), chunk)])
    return out.reshape(x.shape)


# ---- staged PINN identification (Section 5) --------------------------------------------
def _phys_dict(q):
    return {k[3:]: float(jnp.exp(v)) for k, v in q.items() if k.startswith('log')}


def _report(comps):
    return lambda q: ' '.join(f'{k} {v:.5f}' for k, v in _phys_dict(comps.full(q)).items())


def run_stages(obs, tc=TrainConfig(), seed=cfg.SEED, pts_seed=None, stage_tc=None, init=None,
               physics=phys_dose_linear, net=None, pts=None, label='causal'):
    """Stages 2-4 of the curriculum. Returns (history, final params, net).

    stage_tc: optional {'stage2': TrainConfig, ...} overriding tc per stage.
    pts: collocation set; otherwise the paper's, sample_points(seed=pts_seed or seed).
    history[stage] = {'params', 'L', 'minutes', 'trajectory'}.
    """
    stage_tc = stage_tc or {}
    init = {**INIT, **(init or {})}
    lg = {f'log{k}': jnp.log(jnp.float64(v)) for k, v in init.items()}
    net = net or DoseNet()
    if pts is None:
        pts = sample_points(seed=seed if pts_seed is None else pts_seed)
    batch0 = make_dose_batch(pts, collocation_doses(pts, 'zero', seed), untreated(obs))
    batch_all = make_dose_batch(pts, collocation_doses(pts, 'uniform', seed + 1), obs)
    history = {'init': {'params': dict(init)}}

    def run(stage, p0, comps, batch):
        p, info = train(p0, comps, batch, stage_tc.get(stage, tc), label=f'{label} {stage}', report=_report(comps))
        history[stage] = {'params': _phys_dict(comps.full(p)), 'L': info['L'], 'minutes': info['minutes'],
                          'trajectory': info['trajectory']}
        return p

    # stage 2: baseline (D, k) from untreated data; eta has no gradient at a = 0 and is pinned
    c2 = build_dose_comps(physics, net, pin={'logeta': lg['logeta']})
    p2 = run('stage2', {'net': net.init(seed), 'logD': lg['logD'], 'logk': lg['logk']}, c2, batch0)
    # stage 3: eta from all training doses, baseline pinned at its stage-2 values
    c3 = build_dose_comps(physics, net, pin={'logD': p2['logD'], 'logk': p2['logk']})
    p3 = run('stage3', {'net': p2['net'], 'logeta': lg['logeta']}, c3, batch_all)
    # stage 4: joint refinement
    c4 = build_dose_comps(physics, net)
    p4 = run('stage4', {'net': p3['net'], 'logD': p2['logD'], 'logk': p2['logk'], 'logeta': p3['logeta']},
             c4, batch_all)
    return history, p4, net


# ---- reference estimator on the same stages ----------------------------------------------
def _dose_predict(obs, solver, amp=cfg.C0):
    """lp = {'logD', 'logk', 'logeta'} -> FD concentrations at every observation."""
    x, y, t, a = (np.asarray(obs[k]) for k in ('x_d', 'y_d', 't_d', 'a_d'))
    groups = [(float(d), np.flatnonzero(a == d)) for d in np.unique(a)]
    C_init = gaussian_ic(solver.nx, amp, L=solver.L)

    def predict(lp):
        out = jnp.zeros(len(x))
        for d, idx in groups:
            fr = solver(jnp.exp(lp['logD']), linear_sink(jnp.exp(lp['logk']) * (1.0 - jnp.exp(lp['logeta']) * d)),
                        C_init)
            out = out.at[idx].set(interp_frames(fr, solver.dx, jnp.asarray(x[idx]), jnp.asarray(y[idx]),
                                                jnp.asarray(t[idx]), solver.dt_store, solver.L))
        return out
    return predict


_BOUNDS = {'D': (0.02, None), 'k': (1e-4, 0.5), 'eta': (1e-3, 0.999)}


def fit_reference(obs, solver, noise_sd=cfg.NOISE_SD, fixed=None, init=None):
    """Least squares through the FD solver for the free parameters (log scale).

    fixed: {name: value} held at that value (eta is fixed automatically if all data are
    untreated, where it has no effect). Returns params, Laplace SDs/covariance of the
    free log-parameters and the raw fit.
    """
    fixed = dict(fixed or {})
    if np.all(np.asarray(obs['a_d']) == 0.0):
        fixed.setdefault('eta', ETA_INIT)
    init = {**INIT, **(init or {})}
    free = [p for p in PARAMS if p not in fixed]
    base = _dose_predict(obs, solver)
    fixed_log = {f'log{p}': jnp.log(jnp.float64(v)) for p, v in fixed.items()}
    predict = lambda th: base({**fixed_log, **{f'log{p}': th[i] for i, p in enumerate(free)}})
    ub = {'D': 0.97 * solver.D_cap}
    lb_ = np.log([_BOUNDS[p][0] for p in free])
    ub_ = np.log([ub.get(p, _BOUNDS[p][1]) for p in free])
    r = ref_fit(predict, np.log([init[p] for p in free]), np.asarray(obs['C_d']), noise_sd, lb_, ub_)
    sd = np.sqrt(np.diag(r['cov']))
    params = {**fixed, **{p: float(np.exp(r['theta'][i])) for i, p in enumerate(free)}}
    return {'params': params, 'free': free, 'sd_log': dict(zip(free, sd.tolist())),
            'cov_log': r['cov'].tolist(), 'corr': (r['cov'] / np.outer(sd, sd)).tolist(), 'fit': r}


def reference_stages(obs, solver, noise_sd=cfg.NOISE_SD):
    """The reference estimator through the same curriculum as the PINN."""
    s2 = fit_reference(untreated(obs), solver, noise_sd)
    s3 = fit_reference(obs, solver, noise_sd, fixed={'D': s2['params']['D'], 'k': s2['params']['k']})
    s4 = fit_reference(obs, solver, noise_sd, init=s3['params'])
    return {'stage2': s2, 'stage3': s3, 'stage4': s4}


def identifiability(obs, solver, noise_sd=cfg.NOISE_SD, lam=None):
    """Jacobian of the observations w.r.t. (log D, log k, log eta) at lam (default truth).

    For the untreated subset and the full design: column norms (a zero column is
    structurally unidentifiable), condition number of the identifiable columns after
    normalisation, and the Cramer-Rao SDs of the log-parameters.
    """
    lam = lam or TRUE
    th = jnp.log(jnp.asarray([lam[p] for p in PARAMS]))
    out = {}
    for name, o in (('untreated_only', untreated(obs)), ('all_training_doses', obs)):
        base = _dose_predict(o, solver)
        J = np.asarray(jax.jacfwd(lambda t: base({f'log{p}': t[i] for i, p in enumerate(PARAMS)}))(th))
        norms = np.linalg.norm(J, axis=0)
        ok = norms > 1e-10 * norms.max()
        Jn = J[:, ok] / norms[ok]
        sv = np.linalg.svd(Jn, compute_uv=False)
        crlb = np.full(len(PARAMS), np.inf)
        crlb[ok] = noise_sd * np.sqrt(np.diag(np.linalg.inv(J[:, ok].T @ J[:, ok])))
        out[name] = {'column_norms': dict(zip(PARAMS, norms.tolist())),
                     'unidentifiable': [p for p, g in zip(PARAMS, ok) if not g],
                     'condition_number': float(sv[0] / sv[-1]),
                     'crlb_sd_log': {p: (float(v) if np.isfinite(v) else None) for p, v in zip(PARAMS, crlb)}}
    return out


# ---- effects (Section 6) -------------------------------------------------------------
def _cv_weights(solver):
    w = np.ones(solver.nx)
    w[0] = w[-1] = 0.5
    return np.outer(w, w) * solver.dx**2 * solver.mask / solver.L**2   # sums to 1 on the open square


def _trapz(y, x, axis=0):
    return (getattr(np, 'trapezoid', None) or np.trapz)(y, x, axis=axis)


def effect_summaries(C0f, CAf, ts, solver):
    """ATE(t), CE(x, y), GTE, peak effect from fields on the solver nodes, shape (nt, nx, nx)."""
    tau = CAf - C0f
    W = _cv_weights(solver)
    ate = np.sum(tau * W, axis=(1, 2))
    return {'tau': tau, 'ATE': ate, 'CE': _trapz(tau, ts), 'GTE': float(_trapz(ate, ts) / ts[-1]),
            'peak_tau': float(tau.max()), 'peak_ATE': float(ate.max()), 't_peak_ATE': float(ts[np.argmax(ate)])}


def _calibrated_fields(lam, a, solver, frame_idx):
    """Calibrated mechanistic PDE: FD solution at fitted lam for doses 0 and a."""
    f0 = np.asarray(truth_frames(0.0, solver, lam['D'], lam['k'], lam['eta']))[frame_idx]
    fa = np.asarray(truth_frames(a, solver, lam['D'], lam['k'], lam['eta']))[frame_idx]
    return f0, fa


def delta_method_effects(ref_fit_, a_treat, solver, ts, doses):
    """Laplace (delta-method) SDs of ATE_a(t) and GTE over a dose grid, from a reference fit.

    These are the information limit of the design for the effect estimates.
    """
    free = ref_fit_['free']
    cov = np.asarray(ref_fit_['cov_log'])
    lam = ref_fit_['params']
    W = jnp.asarray(_cv_weights(solver))
    C_init = gaussian_ic(solver.nx, L=solver.L)
    idx = np.round(ts / solver.dt_store).astype(int)

    def ate_curve(th, a):
        lp = {f'log{p}': jnp.log(jnp.float64(lam[p])) for p in PARAMS}
        lp.update({f'log{p}': th[i] for i, p in enumerate(free)})
        k = lambda d: jnp.exp(lp['logk']) * (1.0 - jnp.exp(lp['logeta']) * d)
        fr0 = solver(jnp.exp(lp['logD']), linear_sink(k(0.0)), C_init)[idx]
        fra = solver(jnp.exp(lp['logD']), linear_sink(k(a)), C_init)[idx]
        return jnp.sum((fra - fr0) * W, axis=(1, 2))

    th = jnp.log(jnp.asarray([lam[p] for p in free]))
    trap = lambda y: jnp.sum(0.5 * (y[1:] + y[:-1]) * jnp.diff(jnp.asarray(ts))) / ts[-1]
    J_ate = np.asarray(jax.jacfwd(ate_curve)(th, a_treat))
    sd_ate = np.sqrt(np.einsum('ti,ij,tj->t', J_ate, cov, J_ate))
    gte_sd = []
    for d in doses:
        g = np.asarray(jax.jacfwd(lambda t: trap(ate_curve(t, d)))(th))   # forward mode: no tape
        gte_sd.append(float(np.sqrt(g @ cov @ g)))
    return {'sd_ATE': sd_ate, 'sd_GTE_dose_grid': gte_sd}


def evaluate_counterfactual(net, p, lam_pinn, solver, ref=None, a_treat=A_HELDOUT, dt_eval=1.0,
                            doses=np.linspace(0.0, 1.0, 11), h_dose=0.01):
    """Stage 5 at the held-out dose on the solver nodes every dt_eval ms.

    Three estimates of the effect, each against the truth: the network (tau_theta), the
    calibrated PDE at the PINN parameters, and (if ref is given) the calibrated PDE at the
    reference-estimator parameters with delta-method SDs.
    """
    ts = np.arange(0.0, solver.T_end + 1e-9, dt_eval)
    fidx = np.round(ts / solver.dt_store).astype(int)
    x = np.linspace(-solver.L / 2, solver.L / 2, solver.nx)
    Tg, X, Y = np.meshgrid(ts, x, x, indexing='ij')
    rel = lambda e, r: float(100 * np.linalg.norm(e - r) / np.linalg.norm(r))

    truth = {a: np.asarray(truth_frames(a, solver))[fidx] for a in (0.0, a_treat, a_treat - h_dose, a_treat + h_dose)}
    true = effect_summaries(truth[0.0], truth[a_treat], ts, solver)
    C0_n = predict_dose(net, p, X, Y, Tg, 0.0)
    CA_n = predict_dose(net, p, X, Y, Tg, a_treat)
    pinn = effect_summaries(C0_n, CA_n, ts, solver)
    cal = effect_summaries(*_calibrated_fields(lam_pinn, a_treat, solver, fidx), ts, solver)

    def scores(est):
        return {'E_L2_tau_pct': rel(est['tau'], true['tau']),
                'E_Linf_tau': float(np.max(np.abs(est['tau'] - true['tau']))),
                'E_L2_ATE_pct': rel(est['ATE'], true['ATE']),
                'E_L2_CE_pct': rel(est['CE'], true['CE']),
                'GTE': est['GTE'], 'GTE_err_pct': 100 * (est['GTE'] - true['GTE']) / true['GTE'],
                'peak_tau': est['peak_tau'], 'peak_ATE': est['peak_ATE'], 't_peak_ATE': est['t_peak_ATE']}

    i_mid = len(ts) // 2
    S_true = (truth[a_treat + h_dose][i_mid] - truth[a_treat - h_dose][i_mid]) / (2 * h_dose)
    S_pinn = dose_sensitivity(net, p, X[i_mid], Y[i_mid], ts[i_mid], a_treat)

    metrics = {
        'truth': {k: true[k] for k in ('GTE', 'peak_tau', 'peak_ATE', 't_peak_ATE')},
        'pinn_network': {**scores(pinn), 'E_L2_C0_pct': rel(C0_n, truth[0.0]),
                         'E_L2_Ca_pct': rel(CA_n, truth[a_treat]),
                         'S_A_t': float(ts[i_mid]), 'E_L2_S_A_pct': rel(S_pinn, S_true)},
        'pinn_calibrated_pde': scores(cal),
    }
    curves = {'ts': ts, 'x': x, 'true': true, 'pinn': pinn, 'calibrated': cal, 'S_true': S_true, 'S_pinn': S_pinn,
              'C0_pinn': C0_n, 'Ca_pinn': CA_n, 'C0_true': truth[0.0], 'Ca_true': truth[a_treat]}

    # dose-response of GTE
    def gte_of(f0, fa):
        return effect_summaries(f0, fa, ts, solver)['GTE']
    dr = {'doses': doses.tolist(), 'truth': [], 'pinn_network': [], 'pinn_calibrated_pde': []}
    for d in doses:
        dr['truth'].append(gte_of(truth[0.0], np.asarray(truth_frames(d, solver))[fidx]))
        dr['pinn_network'].append(gte_of(C0_n, predict_dose(net, p, X, Y, Tg, d)))
        dr['pinn_calibrated_pde'].append(gte_of(*_calibrated_fields(lam_pinn, d, solver, fidx)))

    if ref is not None:
        lam_ref = ref['params']
        rc = effect_summaries(*_calibrated_fields(lam_ref, a_treat, solver, fidx), ts, solver)
        dm = delta_method_effects(ref, a_treat, solver, ts, doses)
        metrics['reference_calibrated_pde'] = {**scores(rc), 'GTE_sd': dm['sd_GTE_dose_grid'][
            int(np.argmin(np.abs(doses - a_treat)))]}
        curves['reference'] = {**rc, 'sd_ATE': dm['sd_ATE']}
        dr['reference_calibrated_pde'] = [gte_of(*_calibrated_fields(lam_ref, d, solver, fidx)) for d in doses]
        dr['reference_sd'] = dm['sd_GTE_dose_grid']
    return metrics, dr, curves


# ---- one complete study ----------------------------------------------------------------
def pct_errors(params):
    return {p: float(100.0 * (params[p] - TRUE[p]) / TRUE[p]) for p in PARAMS}


def run_study(tc=TrainConfig(), obs_seed=OBS_SEED, seed=cfg.SEED, pts_seed=None, n_obs=cfg.N_OBS,
              noise_sd=cfg.NOISE_SD, stage_tc=None, solver=None, with_reference=True, verbose=True):
    """Stages 1-5 for one observation realization. Returns (results dict, private objects)."""
    solver = solver or make_fd_solver()
    t0 = time.time()
    obs = dose_observations(solver, seed=obs_seed, n_obs=n_obs, noise_sd=noise_sd)
    out = {'design': {'A_train': list(A_TRAIN), 'A_heldout': A_HELDOUT, 'n_obs_per_dose': n_obs,
                      'noise_sd': noise_sd, 'obs_seed': obs_seed, 'net_seed': seed,
                      'pts_seed': seed if pts_seed is None else pts_seed},
           'truth': TRUE, 'init': INIT,
           'identifiability': identifiability(obs, solver, max(noise_sd, 1e-6))}
    ref = None
    if with_reference:
        rs = reference_stages(obs, solver, max(noise_sd, 1e-6))
        ref = rs['stage4']
        out['reference'] = {s: {'params': r['params'], 'err_pct': pct_errors(r['params']), 'sd_log': r['sd_log'],
                                'corr': r['corr'], 'chi2_p': r['fit']['p_chi2']} for s, r in rs.items()}
        if verbose:
            print('  reference stage 4:', {p: f'{e:+.2f}%' for p, e in out['reference']['stage4']['err_pct'].items()})
    history, p, net = run_stages(obs, tc, seed=seed, pts_seed=pts_seed, stage_tc=stage_tc)
    out['pinn'] = {s: {**h, 'err_pct': pct_errors({**INIT, **h['params']})} for s, h in history.items()
                   if s != 'init'}
    metrics, dr, curves = evaluate_counterfactual(net, p, history['stage4']['params'], solver, ref)
    out['stage5'] = metrics
    out['dose_response'] = dr
    out['minutes'] = (time.time() - t0) / 60
    if verbose:
        print('  PINN stage 4:', {q: f'{e:+.2f}%' for q, e in out['pinn']['stage4']['err_pct'].items()})
        for name, m in metrics.items():
            if 'E_L2_tau_pct' in m:
                print(f'  stage 5 {name:<26} tau L2 {m["E_L2_tau_pct"]:7.2f}%  GTE err {m["GTE_err_pct"]:+7.2f}%')
    return out, {'obs': obs, 'net': net, 'p': p, 'curves': curves, 'solver': solver}


# ---- figures ---------------------------------------------------------------------------
def plot_study(res, priv, out_dir):
    """Matched heat maps, effect summaries and parameter recovery (PNG)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv, a = priv['curves'], A_HELDOUT
    ts, x = cv['ts'], cv['x']
    ext = [x[0], x[-1], x[0], x[-1]]

    idxs = [int(np.argmin(np.abs(ts - t))) for t in (5.0, 20.0, 45.0)]
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 14.5))
    for c, i in enumerate(idxs):
        vmax = max(cv['C0_pinn'][i].max(), cv['Ca_pinn'][i].max())
        vt = np.abs(cv['true']['tau'][i]).max()
        rows = [(cv['C0_pinn'][i], f'$C_\\theta^{{(0)}}$', 'viridis', 0, vmax),
                (cv['Ca_pinn'][i], f'$C_\\theta^{{({a})}}$', 'viridis', 0, vmax),
                (cv['pinn']['tau'][i], f'$\\tau_{{\\theta,{a}}}$ (network)', 'RdBu_r', -vt, vt),
                (cv['true']['tau'][i], f'$\\tau_{{{a}}}$ (truth)', 'RdBu_r', -vt, vt)]
        for r, (F, title, cmap, lo, hi) in enumerate(rows):
            ax = axes[r, c]
            im = ax.imshow(F.T, origin='lower', extent=ext, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f'{title}, t = {ts[i]:g} ms')
            ax.set_xlabel('x (µm)')
            ax.set_ylabel('y (µm)')
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Held-out mechanistic counterfactual, do(A = {a})')
    fig.tight_layout()
    fig.savefig(out_dir / 'counterfactual_heatmaps.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    ax.plot(ts, cv['true']['ATE'], 'k-', label='truth')
    ax.plot(ts, cv['pinn']['ATE'], 'r--', label='PINN network')
    ax.plot(ts, cv['calibrated']['ATE'], 'b:', label='PDE at PINN λ̂')
    if 'reference' in cv:
        m, s = cv['reference']['ATE'], cv['reference']['sd_ATE']
        ax.fill_between(ts, m - 1.96 * s, m + 1.96 * s, color='0.8', label='reference ±1.96 SD')
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('$ATE_a(t)$ (µM)')
    ax.set_title(f'Domain-averaged effect, A = {a}')
    ax.legend(fontsize=8)

    ax = axes[1]
    j = len(x) // 2
    ax.plot(x, cv['true']['CE'][:, j], 'k-', label='truth')
    ax.plot(x, cv['pinn']['CE'][:, j], 'r--', label='PINN network')
    ax.plot(x, cv['calibrated']['CE'][:, j], 'b:', label='PDE at PINN λ̂')
    ax.set_xlabel('x (µm), y = 0')
    ax.set_ylabel('$CE_a$ (µM ms)')
    ax.set_title('Cumulative effect, centre line')
    ax.legend(fontsize=8)

    ax = axes[2]
    dr = res['dose_response']
    d = np.asarray(dr['doses'])
    ax.plot(d, dr['truth'], 'k-', label='truth')
    ax.plot(d, dr['pinn_network'], 'r--', label='PINN network')
    ax.plot(d, dr['pinn_calibrated_pde'], 'b:', label='PDE at PINN λ̂')
    if 'reference_calibrated_pde' in dr:
        m, s = np.asarray(dr['reference_calibrated_pde']), np.asarray(dr['reference_sd'])
        ax.fill_between(d, m - 1.96 * s, m + 1.96 * s, color='0.8', label='reference ±1.96 SD')
    for at in A_TRAIN:
        ax.axvline(at, color='0.85', lw=0.8, zorder=0)
    ax.axvline(a, color='orange', ls='--', lw=1, label='held-out dose')
    ax.set_xlabel('dose A')
    ax.set_ylabel('$GTE_a$ (µM)')
    ax.set_title('Dose-response of the global effect')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'effect_summaries.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    stages = ['stage2', 'stage3', 'stage4']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, q in zip(axes, PARAMS):
        ax.plot(range(4), [INIT[q]] + [res['pinn'][s]['params'].get(q, INIT[q]) for s in stages], 'ro-', label='PINN')
        if 'reference' in res:
            ax.plot(range(1, 4), [res['reference'][s]['params'][q] for s in stages], 'bs--', label='reference')
        ax.axhline(TRUE[q], color='k', ls=':', label='truth')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['init'] + stages)
        ax.set_title(q)
        ax.legend(fontsize=8)
    fig.suptitle('Parameter recovery through the curriculum')
    fig.tight_layout()
    fig.savefig(out_dir / 'parameter_recovery.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
