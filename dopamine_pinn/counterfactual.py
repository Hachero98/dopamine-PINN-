"""Counterfactual Cramer-Rao bound and the reference estimator on the exact linear model.

Paper 2, Part A (work plan tasks T2 and A1). The model is the dose-aware linear model of
`causal`,

    dC/dt = D lap C - k (1 - eta A) C  on [-L/2, L/2]^2, zero flux, Gaussian release,

whose exact solution under do(A = a) is the image series of paper 1 with uptake rate
k_a = k (1 - eta a):

    C^(a)(x, y, t) = C0 sigma^2 / v exp(-k_a t) g(x, v) g(y, v),   v = sigma^2 + 2 D t,
    g(x, v) = sum_m exp(-(x - m L)^2 / (2 v)).

Here it is both the truth and the fitted model, as in paper 1's review analyses; the FD
solver used by the PINN studies agrees with it to 0.06% (results/exact_solution_check.json).

Counterfactual Cramer-Rao bound (T2). theta = (log D, log k, log eta), Fisher information
of all training doses F = J^T J / sigma_n^2, J = dC_obs/dtheta at the truth. For any
smooth effect functional g(theta) (GTE_a, ATE_a(t), tau_a(x, y, t)) and any unbiased
estimator,  Var(g_hat) >= grad g^T F^{-1} grad g.  For a field, the expected squared
L2 error is bounded by trace(F^{-1} G^T G), G the Jacobian of the field on the grid.
The bound depends on the design through F only; grad g depends only on the truth.

Remark. With zero-flux walls the domain mean of C^(a) is m0 exp(-k_a t) for every D, so
ATE_a(t) and GTE_a do not depend on D at all: their bound involves (k, eta) only. The
field tau_a does depend on D. (On the node grid the trapezoid weights give a D-gradient
of about 1e-6 of the k-gradient.)

Observation designs copy `observations.scattered` draw for draw (locations, times, then
noise, dose j from seed + 100 j), so a seed gives the same points and noise as the FD
runs of `causal.dose_observations`.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
from scipy.stats import beta

from . import config as cfg
from .causal import A_TRAIN, A_HELDOUT, DOSE_SEED_STEP, OBS_SEED, PARAMS, TRUE, INIT

N_IMAGES = 10                     # images per direction (paper 1: 10 and 14 agree)
LOG_LB = np.log([0.02, 1e-4, 1e-3])
LOG_UB = np.log([5.0, 0.5, 0.999])
TH_TRUE = np.log([TRUE[p] for p in PARAMS])


# ---- exact potential outcomes -----------------------------------------------------------
def _g(x, v, L=cfg.L):
    m = jnp.arange(-N_IMAGES, N_IMAGES + 1) * L
    return jnp.sum(jnp.exp(-(x[..., None] - m) ** 2 / (2.0 * v[..., None])), axis=-1)


def exact_C(D, k, x, y, t, sigma=cfg.SIGMA, C0=cfg.C0, L=cfg.L):
    """Exact solution of the bounded linear problem at points (x, y, t) broadcast together."""
    x, y, t = jnp.broadcast_arrays(jnp.asarray(x), jnp.asarray(y), jnp.asarray(t))
    v = sigma**2 + 2.0 * D * t
    return C0 * sigma**2 / v * jnp.exp(-k * t) * _g(x, v, L) * _g(y, v, L)


def k_dose(th, a):
    return jnp.exp(th[1]) * (1.0 - jnp.exp(th[2]) * a)


def model_obs(th, x, y, t, a):
    """theta = (log D, log k, log eta) -> exact concentrations at observations with doses a."""
    return exact_C(jnp.exp(th[0]), k_dose(th, a), x, y, t)


_model = jax.jit(model_obs)
_jac = jax.jit(jax.jacfwd(model_obs))


# ---- observation design (same draws as observations.scattered) ----------------------------
def exact_observations(seed=OBS_SEED, doses=A_TRAIN, n_obs=cfg.N_OBS, noise_sd=cfg.NOISE_SD,
                       th=TH_TRUE, t_window=cfg.T_WINDOW, L=cfg.L):
    assert A_HELDOUT not in doses, 'the held-out dose must never be observed'
    parts = []
    for j, a in enumerate(doses):
        rng = np.random.default_rng(seed + DOSE_SEED_STEP * j)
        ox = rng.uniform(-L / 2, L / 2, n_obs)
        oy = rng.uniform(-L / 2, L / 2, n_obs)
        ot = rng.uniform(*t_window, n_obs)
        clean = np.asarray(_model(jnp.asarray(th), ox, oy, ot, float(a)))
        parts.append({'x_d': ox, 'y_d': oy, 't_d': ot, 'a_d': np.full(n_obs, float(a)),
                      'C_clean': clean, 'C_d': clean + rng.normal(0.0, noise_sd, n_obs)})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _xyta(obs):
    return tuple(jnp.asarray(obs[k]) for k in ('x_d', 'y_d', 't_d', 'a_d'))


def jacobian(obs, th=TH_TRUE):
    return np.asarray(_jac(jnp.asarray(th), *_xyta(obs)))


def fisher(obs, th=TH_TRUE, noise_sd=cfg.NOISE_SD):
    """Fisher information of (log D, log k, log eta) from all observations."""
    J = jacobian(obs, th)
    return J.T @ J / noise_sd**2


# ---- reference estimator ---------------------------------------------------------------
def fit_exact(obs, noise_sd=cfg.NOISE_SD, init=INIT):
    """Least squares of (log D, log k, log eta) with the exact model; Laplace covariance."""
    xyta = _xyta(obs)
    C = np.asarray(obs['C_d'])
    th0 = np.clip(np.log([init[p] for p in PARAMS]), LOG_LB + 1e-9, LOG_UB - 1e-9)
    r = least_squares(lambda th: np.asarray(_model(jnp.asarray(th), *xyta)) - C, th0,
                      jac=lambda th: np.asarray(_jac(jnp.asarray(th), *xyta)),
                      bounds=(LOG_LB, LOG_UB), x_scale='jac', xtol=1e-12, ftol=1e-12, gtol=1e-12)
    cov = np.linalg.inv(r.jac.T @ r.jac) * noise_sd**2
    return {'theta': r.x, 'cov': cov, 'params': dict(zip(PARAMS, np.exp(r.x).tolist())),
            'chi2': float(np.sum(r.fun**2) / noise_sd**2), 'nfev': int(r.nfev),
            'at_bound': bool(np.any(np.isclose(r.x, LOG_LB) | np.isclose(r.x, LOG_UB)))}


# ---- effect functionals on the evaluation grid -------------------------------------------
class EffectGrid:
    """Solver nodes (nx per side) every dt ms on [0, T]; the weights and summaries of
    causal.effect_summaries (trapezoid on the open square, normalised to 1)."""

    def __init__(self, nx=cfg.NX_FD, dt=1.0, T=cfg.T, L=cfg.L):
        self.x = np.linspace(-L / 2, L / 2, nx)
        self.ts = np.arange(0.0, T + 1e-9, dt)
        w = np.ones(nx)
        w[0] = w[-1] = 0.5
        self.w = w / w.sum()                      # 1-D weights; 2-D weights are w_i w_j

    def field(self, th, a):
        """C^(a) on the grid, shape (nt, nx, nx), separable in x and y."""
        D, ka = jnp.exp(th[0]), k_dose(th, a)
        t = jnp.asarray(self.ts)[:, None]
        v = cfg.SIGMA**2 + 2.0 * D * t
        gx = _g(jnp.asarray(self.x)[None, :], v)            # (nt, nx)
        amp = cfg.C0 * cfg.SIGMA**2 / v[:, 0] * jnp.exp(-ka * t[:, 0])
        return amp[:, None, None] * gx[:, :, None] * gx[:, None, :]

    def tau(self, th, a):
        return self.field(th, a) - self.field(th, 0.0)

    def ate(self, th, a):
        w = jnp.asarray(self.w)
        return jnp.einsum('tij,i,j->t', self.tau(th, a), w, w)

    def gte(self, th, a):
        y, ts = self.ate(th, a), jnp.asarray(self.ts)
        return jnp.sum(0.5 * (y[1:] + y[:-1]) * jnp.diff(ts)) / ts[-1]

    def gradients(self, a, th=TH_TRUE):
        """Values and Jacobians w.r.t. theta of GTE_a, ATE_a(t) and tau_a at th."""
        th = jnp.asarray(th)
        out = {}
        for name, f in (('GTE', self.gte), ('ATE', self.ate), ('tau', self.tau)):
            out[name] = np.asarray(f(th, a))
            out['d' + name] = np.asarray(jax.jacfwd(f)(th, a))
        return out


def gte_closed_form(a, k=cfg.K_TRUE, eta=TRUE['eta'], T=cfg.T, grid=None):
    """GTE_a = (m0 / T) [(1 - e^{-k_a T}) / k_a - (1 - e^{-k T}) / k], independent of D.

    m0 is the domain mean of the release (grid weights). Exact in continuous time; the
    grid's trapezoid rule in t differs by O(dt^2).
    """
    grid = grid or EffectGrid()
    C_init = np.asarray(grid.field(jnp.asarray(TH_TRUE), 0.0)[0])
    m0 = float(grid.w @ C_init @ grid.w)
    ka = k * (1.0 - eta * a)
    return m0 / T * ((1.0 - np.exp(-ka * T)) / ka - (1.0 - np.exp(-k * T)) / k)


# ---- counterfactual Cramer-Rao bound (T2) ---------------------------------------------------
def counterfactual_crlb(F, grads):
    """Bounds from Fisher information F and the gradients of EffectGrid.gradients.

    Returns SDs of the parameters (log scale), of GTE and ATE(t), the pointwise SD map of
    tau, and the bound on the root expected squared L2 error of tau and ATE.
    """
    Fi = np.linalg.inv(F)
    g = grads['dGTE']
    J_ate = grads['dATE']
    G_tau = grads['dtau'].reshape(-1, 3)
    return {'sd_log_params': dict(zip(PARAMS, np.sqrt(np.diag(Fi)).tolist())),
            'sd_GTE': float(np.sqrt(g @ Fi @ g)),
            'sd_ATE': np.sqrt(np.einsum('ti,ij,tj->t', J_ate, Fi, J_ate)),
            'sd_tau': np.sqrt(np.einsum('ni,ij,nj->n', G_tau, Fi, G_tau)).reshape(grads['tau'].shape),
            'rmse_L2_tau': float(np.sqrt(np.trace(Fi @ (G_tau.T @ G_tau)))),
            'rmse_L2_ATE': float(np.sqrt(np.trace(Fi @ (J_ate.T @ J_ate))))}


def delta_sd(cov, grad):
    """Delta-method SD of a scalar (grad shape (3,)) or each entry of an array (..., 3)."""
    return np.sqrt(np.einsum('...i,ij,...j->...', grad, cov, grad))


# ---- statistics ----------------------------------------------------------------------
def clopper_pearson(k, n, level=0.95):
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def summarise(errors, sds=None, covered=None):
    """Bias (with its SE), empirical SD, RMSE, mean predicted SD and coverage of a sample."""
    e = np.asarray(errors, float)
    n = e.size
    out = {'n': int(n), 'bias': float(e.mean()), 'bias_se': float(e.std(ddof=1) / np.sqrt(n)),
           'sd': float(e.std(ddof=1)), 'rmse': float(np.sqrt(np.mean(e**2))), 'median': float(np.median(e))}
    if sds is not None:
        out['rms_predicted_sd'] = float(np.sqrt(np.mean(np.asarray(sds, float) ** 2)))
    if covered is not None:
        c = int(np.sum(covered))
        out['covered'] = c
        out['coverage'] = c / n
        out['coverage_ci95'] = clopper_pearson(c, n)
    return out
