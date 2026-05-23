"""
dopamine_PINN.py
================
Physics-Informed Neural Network solver for dopamine diffusion in
the synaptic cleft, corresponding to paper B3:
    "Physics-Informed Neural Networks for Modeling Dopamine
     Neurotransmitter Diffusion in Synaptic Clefts: A Computational
     Approach to Parkinson's Disease Dynamics"

Implementation: pure JAX with Flax NNX (network), Optax (Adam),
and jaxopt (L-BFGS). JIT compilation makes the training step
2 to 5x faster than the DeepXDE/PyTorch baseline preserved in
dopamine_PINN_deepxde.{py,ipynb} for the same hyperparameters.

GOVERNING PDE (2D reaction-diffusion):
    dC/dt = D ( d^2 C/dx^2 + d^2 C/dy^2 ) - k C
        (x, y) in (-L/2, L/2)^2,  t in [0, T]

INITIAL CONDITION (radial Gaussian release pulse at t = 0):
    C(x, y, 0) = C0 exp( -(x^2 + y^2) / (2 sigma^2) )

BOUNDARY CONDITIONS (zero-flux / Neumann on all four edges):
    grad C . n = 0     on dOmega

ANALYTICAL SOLUTION (infinite-domain, valid while the pulse has
not reached +/- L/2):
    C(x, y, t) = C0 sigma^2 / (sigma^2 + 2 D t)
                 * exp( -(x^2 + y^2) / (2 (sigma^2 + 2 D t)) )
                 * exp( -k t )

HOW TO RUN
----------
Local (GPU optional):
    pip install -r requirements.txt
    python dopamine_PINN.py

Colab (recommended, GPU runtime):
    !pip install -q jax flax optax jaxopt matplotlib scipy
    !python dopamine_PINN.py

Outputs are written to ./figures/ and printed to stdout.
Random seeds are fixed for reproducibility.
"""

# =============================================================
# 0. Imports and reproducibility
# =============================================================
import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from jaxopt import LBFGS

# Use float64 for inverse-problem numerical stability
jax.config.update("jax_enable_x64", True)

SEED = 1234
np.random.seed(SEED)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# =============================================================
# 1. Physical parameters (striatal dopamine, Cragg & Rice 2004;
#    Nicholson & Phillips 1981; Wiencke et al. 2020)
# =============================================================
D_TRUE = 0.32     # mu_m^2 / ms     diffusion coefficient
K_TRUE = 0.05     # 1 / ms          linearised reuptake rate
L      = 5.0      # mu_m            domain side length, [-L/2, L/2]^2
T      = 20.0     # ms              simulation time
SIGMA  = 0.5      # mu_m            Gaussian release pulse width
C0     = 1.0      # mu_M            peak release concentration

# Numerical / training hyperparameters
N_DOMAIN    = 10_000
N_BOUNDARY  = 400          # 100 per edge x 4 edges
N_INITIAL   = 400          # over the 2D square
N_TEST      = 5_000
LAYERS      = [3, 64, 64, 64, 64, 1]   # (x, y, t) -> C
ADAM_ITERS  = 20_000
LBFGS_ITERS = 5_000
LR          = 1e-3

# Inverse-problem observation count.
# The 2D square at the per-area density used for 1D requires 400
# observations (4x scaling from a 1D segment). Sparser sampling
# produces an underdetermined inverse problem with biased recovery.
N_OBS = 400


# =============================================================
# 2. Analytical reference solution (Gaussian IC, infinite-domain
#    approximation, valid while the pulse has not reached +/- L/2).
# =============================================================
def C_analytical(x, y, t, D=D_TRUE, k=K_TRUE, sigma=SIGMA, C0=C0):
    s2 = sigma ** 2 + 2.0 * D * t
    amp = C0 * sigma ** 2 / s2
    return amp * np.exp(-(x ** 2 + y ** 2) / (2.0 * s2)) * np.exp(-k * t)


# =============================================================
# 3. Finite-difference reference solver
#    2D explicit, 5-point Laplacian, zero-flux mirror BCs.
# =============================================================
def fd_reference_2d(D=D_TRUE, k=K_TRUE, L=L, T=T, sigma=SIGMA, C0=C0,
                    nx=101, ny=101, nt=None,
                    snapshot_times=(1.0, 5.0, 10.0, 20.0)):
    """Explicit 2D FD solver.

    Returns (x, y, t_snapshots, snapshots) where snapshots is a dict
    {t: C[ny, nx]} for the requested times, plus the time vector.
    Auto-selects nt to satisfy dt <= 0.95 / (2 D (1/dx^2 + 1/dy^2)).
    """
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    if nt is None:
        dt_max = 0.95 / (2.0 * D * (1.0 / dx ** 2 + 1.0 / dy ** 2))
        nt = int(np.ceil(T / dt_max)) + 1
    dt = T / (nt - 1)
    assert dt <= 1.0 / (2.0 * D * (1.0 / dx ** 2 + 1.0 / dy ** 2)), \
        "FD stability violated; refine grids."

    x = np.linspace(-L / 2, L / 2, nx)
    y = np.linspace(-L / 2, L / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    C = C0 * np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))

    ax = D * dt / dx ** 2
    ay = D * dt / dy ** 2

    snapshots = {}
    snapset = set(float(s) for s in snapshot_times)
    t_arr = np.linspace(0.0, T, nt)
    if 0.0 in snapset:
        snapshots[0.0] = C.copy()

    for n in range(1, nt):
        lap = np.zeros_like(C)
        # d^2 C / dx^2  (axis 1 is x in xy meshgrid)
        lap[:, 1:-1] += C[:, 2:] - 2.0 * C[:, 1:-1] + C[:, :-2]
        lap[:, 0]    += 2.0 * (C[:, 1]  - C[:, 0])
        lap[:, -1]   += 2.0 * (C[:, -2] - C[:, -1])
        Lx = lap * (D * dt / dx ** 2)
        lap2 = np.zeros_like(C)
        # d^2 C / dy^2  (axis 0 is y)
        lap2[1:-1, :] += C[2:, :] - 2.0 * C[1:-1, :] + C[:-2, :]
        lap2[0, :]    += 2.0 * (C[1, :]  - C[0, :])
        lap2[-1, :]   += 2.0 * (C[-2, :] - C[-1, :])
        Ly = lap2 * (D * dt / dy ** 2)
        C = C + Lx + Ly - k * dt * C
        t_now = t_arr[n]
        for st in snapset:
            if st not in snapshots and abs(t_now - st) <= dt / 2.0:
                snapshots[st] = C.copy()

    return x, y, t_arr, snapshots


# =============================================================
# 4. PINN: Flax NNX MLP
# =============================================================
class MLP(nnx.Module):
    """Fully-connected feedforward net with tanh activations.

    Input: (x, y, t) in R^3. Output: scalar C(x, y, t).

    Each Linear layer is stored as its own attribute (lin_0, lin_1, ...)
    rather than in a Python list, so Flax NNX treats them as named
    submodules without needing nnx.List wrapping (which is not present
    in all Flax versions).
    """
    def __init__(self, layers, *, rngs: nnx.Rngs):
        self.n_layers = len(layers) - 1
        for i in range(self.n_layers):
            setattr(self, f"lin_{i}",
                    nnx.Linear(layers[i], layers[i + 1],
                               kernel_init=nnx.initializers.glorot_normal(),
                               bias_init=nnx.initializers.zeros_init(),
                               param_dtype=jnp.float64,
                               rngs=rngs))

    def __call__(self, xyt):
        h = xyt
        for i in range(self.n_layers - 1):
            h = jnp.tanh(getattr(self, f"lin_{i}")(h))
        return getattr(self, f"lin_{self.n_layers - 1}")(h)


class InverseMLP(nnx.Module):
    """MLP + log-parametrized D and k as learnable scalars.

    log-param keeps D, k strictly positive without box constraints,
    avoiding the catastrophic D < 0 minima found in earlier runs.
    """
    def __init__(self, layers, D0, k0, *, rngs: nnx.Rngs):
        self.mlp = MLP(layers, rngs=rngs)
        self.D_log = nnx.Param(jnp.asarray(np.log(D0), dtype=jnp.float64))
        self.k_log = nnx.Param(jnp.asarray(np.log(k0), dtype=jnp.float64))

    def __call__(self, xyt):
        return self.mlp(xyt)

    @property
    def D(self):
        # .get_value() is the current NNX API; fall back to .value on
        # older Flax versions that don't yet expose the new method.
        getter = getattr(self.D_log, "get_value", None)
        return jnp.exp(getter() if callable(getter) else self.D_log.value)

    @property
    def k(self):
        getter = getattr(self.k_log, "get_value", None)
        return jnp.exp(getter() if callable(getter) else self.k_log.value)


def C_at(model, x, y, t):
    """Scalar concentration prediction at a single point."""
    return model(jnp.stack([x, y, t]))[0]


def _residual_one(model, x, y, t, D, k):
    """PDE residual at one point: dC/dt - D Laplacian(C) + k C."""
    # First derivatives w.r.t. each input
    dC_dx = jax.grad(C_at, argnums=1)(model, x, y, t)
    dC_dy = jax.grad(C_at, argnums=2)(model, x, y, t)
    dC_dt = jax.grad(C_at, argnums=3)(model, x, y, t)
    # Second derivatives for the Laplacian
    d2C_dx2 = jax.grad(lambda xx: jax.grad(C_at, argnums=1)(model, xx, y, t))(x)
    d2C_dy2 = jax.grad(lambda yy: jax.grad(C_at, argnums=2)(model, x, yy, t))(y)
    return dC_dt - D * (d2C_dx2 + d2C_dy2) + k * C_at(model, x, y, t)


def _bc_normal_deriv(model, x, y, t, nx, ny):
    """Normal derivative grad(C) . n at one boundary point."""
    dC_dx = jax.grad(C_at, argnums=1)(model, x, y, t)
    dC_dy = jax.grad(C_at, argnums=2)(model, x, y, t)
    return nx * dC_dx + ny * dC_dy


# =============================================================
# 5. Point sampling (one-shot, fixed throughout training)
# =============================================================
def sample_points(n_domain, n_initial, n_boundary, rng):
    """Sample interior, initial, and boundary collocation points.

    Returns a dict of jnp arrays ready to be fed into the loss.
    """
    # Interior collocation
    x_r = rng.uniform(-L / 2, L / 2, n_domain)
    y_r = rng.uniform(-L / 2, L / 2, n_domain)
    t_r = rng.uniform(0.0, T, n_domain)

    # Initial-condition points (t = 0)
    x_i = rng.uniform(-L / 2, L / 2, n_initial)
    y_i = rng.uniform(-L / 2, L / 2, n_initial)
    C_i = C0 * np.exp(-(x_i ** 2 + y_i ** 2) / (2.0 * SIGMA ** 2))

    # Boundary: n_boundary / 4 points on each edge with outward normal
    per_edge = n_boundary // 4
    edges = []
    # left edge (x = -L/2, normal = (-1, 0))
    edges.append((
        np.full(per_edge, -L / 2),
        rng.uniform(-L / 2, L / 2, per_edge),
        rng.uniform(0.0, T, per_edge),
        np.full(per_edge, -1.0),
        np.full(per_edge, 0.0),
    ))
    # right edge
    edges.append((
        np.full(per_edge, L / 2),
        rng.uniform(-L / 2, L / 2, per_edge),
        rng.uniform(0.0, T, per_edge),
        np.full(per_edge, 1.0),
        np.full(per_edge, 0.0),
    ))
    # bottom edge (y = -L/2, normal = (0, -1))
    edges.append((
        rng.uniform(-L / 2, L / 2, per_edge),
        np.full(per_edge, -L / 2),
        rng.uniform(0.0, T, per_edge),
        np.full(per_edge, 0.0),
        np.full(per_edge, -1.0),
    ))
    # top edge
    edges.append((
        rng.uniform(-L / 2, L / 2, per_edge),
        np.full(per_edge, L / 2),
        rng.uniform(0.0, T, per_edge),
        np.full(per_edge, 0.0),
        np.full(per_edge, 1.0),
    ))
    x_b = np.concatenate([e[0] for e in edges])
    y_b = np.concatenate([e[1] for e in edges])
    t_b = np.concatenate([e[2] for e in edges])
    nx_b = np.concatenate([e[3] for e in edges])
    ny_b = np.concatenate([e[4] for e in edges])

    return {
        "x_r":  jnp.asarray(x_r),
        "y_r":  jnp.asarray(y_r),
        "t_r":  jnp.asarray(t_r),
        "x_i":  jnp.asarray(x_i),
        "y_i":  jnp.asarray(y_i),
        "C_i":  jnp.asarray(C_i),
        "x_b":  jnp.asarray(x_b),
        "y_b":  jnp.asarray(y_b),
        "t_b":  jnp.asarray(t_b),
        "nx_b": jnp.asarray(nx_b),
        "ny_b": jnp.asarray(ny_b),
    }


# =============================================================
# 6. Loss functions
# =============================================================
def _losses(model, pts, D, k):
    """Component losses (residual, IC, BC) shared by forward and inverse."""
    # Residual
    res = jax.vmap(_residual_one, in_axes=(None, 0, 0, 0, None, None))(
        model, pts["x_r"], pts["y_r"], pts["t_r"], D, k
    )
    L_r = jnp.mean(res ** 2)

    # Initial condition: C(x, y, 0) - C_i
    C_pred_ic = jax.vmap(C_at, in_axes=(None, 0, 0, None))(
        model, pts["x_i"], pts["y_i"], jnp.float64(0.0)
    )
    L_i = jnp.mean((C_pred_ic - pts["C_i"]) ** 2)

    # Boundary Neumann
    nd = jax.vmap(_bc_normal_deriv, in_axes=(None, 0, 0, 0, 0, 0))(
        model, pts["x_b"], pts["y_b"], pts["t_b"], pts["nx_b"], pts["ny_b"]
    )
    L_b = jnp.mean(nd ** 2)
    return L_r, L_i, L_b


def forward_loss(model, pts):
    """Forward problem: residual + IC + BC, no data."""
    L_r, L_i, L_b = _losses(model, pts, D_TRUE, K_TRUE)
    # Loss weights match the DeepXDE configuration: [10, 1, 1] for [r, i, b]
    return 10.0 * L_r + 1.0 * L_i + 1.0 * L_b


def inverse_loss(model, pts, obs):
    """Inverse problem: residual + IC + BC + data, learnable D, k."""
    D = model.D
    k = model.k
    L_r, L_i, L_b = _losses(model.mlp, pts, D, k)
    # Data
    C_pred_obs = jax.vmap(C_at, in_axes=(None, 0, 0, 0))(
        model.mlp, obs["x_d"], obs["y_d"], obs["t_d"]
    )
    L_d = jnp.mean((C_pred_obs - obs["C_d"]) ** 2)
    # Loss weights [10, 1, 1, 1] for [r, i, b, d]
    return 10.0 * L_r + 1.0 * L_i + 1.0 * L_b + 1.0 * L_d


# =============================================================
# 7. Forward training
# =============================================================
def train_forward():
    print("\n" + "=" * 60)
    print("  FORWARD PROBLEM - PINN training")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    pts = sample_points(N_DOMAIN, N_INITIAL, N_BOUNDARY, rng)

    rngs = nnx.Rngs(SEED)
    model = MLP(LAYERS, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    @nnx.jit
    def adam_step(model, optimizer, pts):
        loss_val, grads = nnx.value_and_grad(forward_loss)(model, pts)
        optimizer.update(model, grads)
        return loss_val

    t0 = time.time()
    for it in range(ADAM_ITERS):
        loss_val = adam_step(model, optimizer, pts)
        if it % 1000 == 0:
            print(f"  Adam iter {it:>5d}  loss = {float(loss_val):.4e}")
    print(f"  Adam phase: {time.time() - t0:.1f}s")

    # L-BFGS phase via jaxopt: split out a pure parameter pytree
    gdef, state = nnx.split(model)

    def lbfgs_loss(params):
        m = nnx.merge(gdef, params)
        return forward_loss(m, pts)

    t0 = time.time()
    solver = LBFGS(fun=lbfgs_loss, maxiter=LBFGS_ITERS, tol=1e-9)
    result = solver.run(state)
    model = nnx.merge(gdef, result.params)
    print(f"  L-BFGS phase: {time.time() - t0:.1f}s")
    return model


# =============================================================
# 8. Forward evaluation
# =============================================================
def evaluate_forward(model):
    nx, ny, nt = 101, 101, 21
    xs = np.linspace(-L / 2, L / 2, nx)
    ys = np.linspace(-L / 2, L / 2, ny)
    ts = np.linspace(0.0, T, nt)

    snapshot_times = [1.0, 5.0, 10.0, 20.0]
    # FD reference (xy meshgrid: shape [ny, nx])
    x_fd, y_fd, _, fd_snaps = fd_reference_2d(snapshot_times=snapshot_times)

    @jax.jit
    def predict_grid(t_val):
        XX, YY = jnp.meshgrid(jnp.asarray(xs), jnp.asarray(ys), indexing="xy")
        flat = jnp.stack([XX.ravel(), YY.ravel(),
                          jnp.full(XX.size, t_val)], axis=1)
        C = jax.vmap(lambda row: model(row)[0])(flat)
        return C.reshape(ny, nx)

    # Full space-time aggregate against FD
    C_pinn_stack = []
    C_fd_stack   = []
    for ts_val in ts:
        Cp = np.asarray(predict_grid(jnp.float64(ts_val)))
        C_pinn_stack.append(Cp)
        # interpolate FD onto this t using the analytical between FD snapshots
        # is not direct; instead use the FD snapshots dict at matching t.
        # For aggregate metric, evaluate at the FD snapshot times only.
    C_pinn_stack = np.stack(C_pinn_stack, axis=0)   # [nt, ny, nx]

    # Aggregate L2 against FD: compare on the snapshot times
    fd_arr = []
    pp_arr = []
    for st in snapshot_times:
        fd_arr.append(fd_snaps[st])
        idx = int(np.argmin(np.abs(ts - st)))
        pp_arr.append(C_pinn_stack[idx])
    fd_arr = np.stack(fd_arr, axis=0)
    pp_arr = np.stack(pp_arr, axis=0)
    L2_fd = 100.0 * np.linalg.norm(pp_arr - fd_arr) / np.linalg.norm(fd_arr)

    # Analytical aggregate (early-time only meaningful, but we report the
    # full window so users can see the divergence; downstream code reports
    # per-snapshot separately for the manuscript).
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    anal_stack = []
    for st in snapshot_times:
        anal_stack.append(C_analytical(XX, YY, st))
    anal_stack = np.stack(anal_stack, axis=0)
    L2_anal_full = 100.0 * np.linalg.norm(pp_arr - anal_stack) / np.linalg.norm(anal_stack)

    # Per-snapshot
    per_snapshot = {}
    for i, st in enumerate(snapshot_times):
        Cp = pp_arr[i]
        Ca = anal_stack[i]
        Cf = fd_arr[i]
        per_snapshot[f"t_{int(st)}"] = {
            "L2_anal_pct": 100.0 * np.linalg.norm(Cp - Ca) / np.linalg.norm(Ca),
            "L2_fd_pct":   100.0 * np.linalg.norm(Cp - Cf) / np.linalg.norm(Cf),
        }

    return {
        "xs": xs, "ys": ys, "ts": ts,
        "C_pinn_grid":      C_pinn_stack,
        "snapshot_times":   snapshot_times,
        "C_pinn_snaps":     pp_arr,
        "C_anal_snaps":     anal_stack,
        "C_fd_snaps":       fd_arr,
        "L2_vs_analytical_pct": L2_anal_full,
        "L2_vs_fd_pct":         L2_fd,
        "per_snapshot":         per_snapshot,
    }


# =============================================================
# 9. Inverse problem
# =============================================================
def make_noisy_observations(n_obs=N_OBS, noise_pct=2.0, rng=None):
    """Sample observations from the FD reference solver + Gaussian noise."""
    from scipy.interpolate import RegularGridInterpolator
    rng = rng if rng is not None else np.random.default_rng(SEED)

    # Get FD reference on a dense space-time grid for interpolation.
    # We use a coarser time set (snapshots) and interpolate linearly in time
    # between adjacent snapshots, which is sufficient given the slow time
    # variation relative to FD time step.
    x_fd, y_fd, t_fd, fd_snaps = fd_reference_2d(
        snapshot_times=tuple(np.linspace(0.5, T / 2.0, 11)),
    )
    snap_ts = sorted(fd_snaps.keys())
    snap_vol = np.stack([fd_snaps[t] for t in snap_ts], axis=0)  # [n_t, ny, nx]
    interp = RegularGridInterpolator(
        (np.array(snap_ts), y_fd, x_fd), snap_vol,
        bounds_error=False, fill_value=0.0,
    )
    obs_x = rng.uniform(-L / 2, L / 2, n_obs)
    obs_y = rng.uniform(-L / 2, L / 2, n_obs)
    obs_t = rng.uniform(0.5, T / 2.0, n_obs)
    C_clean = interp(np.stack([obs_t, obs_y, obs_x], axis=1))
    noise = rng.normal(0.0, (noise_pct / 100.0) * C0, n_obs)
    return obs_x, obs_y, obs_t, C_clean + noise


def train_inverse(obs_x, obs_y, obs_t, obs_C):
    print("\n" + "=" * 60)
    print("  INVERSE PROBLEM - Recover D and k from noisy data")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    pts = sample_points(N_DOMAIN, N_INITIAL, N_BOUNDARY, rng)
    obs = {
        "x_d": jnp.asarray(obs_x),
        "y_d": jnp.asarray(obs_y),
        "t_d": jnp.asarray(obs_t),
        "C_d": jnp.asarray(obs_C),
    }

    # Deliberately offset initial guesses: D_0 = 0.30, k_0 = 0.04
    rngs = nnx.Rngs(SEED)
    model = InverseMLP(LAYERS, D0=0.30, k0=0.04, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    history = []

    @nnx.jit
    def adam_step(model, optimizer, pts, obs):
        loss_val, grads = nnx.value_and_grad(inverse_loss)(model, pts, obs)
        optimizer.update(model, grads)
        return loss_val

    t0 = time.time()
    for it in range(ADAM_ITERS):
        loss_val = adam_step(model, optimizer, pts, obs)
        if it % 500 == 0:
            D_now = float(model.D)
            k_now = float(model.k)
            history.append((it, D_now, k_now))
        if it % 1000 == 0:
            print(f"  Adam iter {it:>5d}  loss = {float(loss_val):.4e}  "
                  f"D = {float(model.D):.4f}  k = {float(model.k):.4f}")
    print(f"  Adam phase: {time.time() - t0:.1f}s")

    # L-BFGS phase
    gdef, state = nnx.split(model)

    def lbfgs_loss(params):
        m = nnx.merge(gdef, params)
        return inverse_loss(m, pts, obs)

    t0 = time.time()
    solver = LBFGS(fun=lbfgs_loss, maxiter=LBFGS_ITERS, tol=1e-9)
    result = solver.run(state)
    state = result.params
    model = nnx.merge(gdef, state)
    print(f"  L-BFGS phase: {time.time() - t0:.1f}s")

    D_rec = float(model.D)
    k_rec = float(model.k)
    history.append((ADAM_ITERS + LBFGS_ITERS, D_rec, k_rec))

    # Persist history to variables.dat for plotting
    with open(FIG_DIR / "variables.dat", "w") as fh:
        for it, D, k in history:
            fh.write(f"{it}\t[{np.log(D):.6f}, {np.log(k):.6f}]\n")

    return model, D_rec, k_rec


# =============================================================
# 10. Plotting
# =============================================================
def plot_forward(result, savepath):
    snapshot_times = result["snapshot_times"][:3]   # t = 1, 5, 10 ms
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    xs, ys = result["xs"], result["ys"]
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    for col, st in enumerate(snapshot_times):
        idx = result["snapshot_times"].index(st)
        Cp = result["C_pinn_snaps"][idx]
        Ca = result["C_anal_snaps"][idx]
        Cf = result["C_fd_snaps"][idx]
        vmax = max(Cp.max(), Cf.max())
        for row, (field, name) in enumerate(
            [(Ca, "Analytical"), (Cp, "PINN"), (Cf, "FD")]
        ):
            ax = axes[row, col]
            im = ax.imshow(field, origin="lower", extent=extent,
                           vmin=0, vmax=vmax, cmap="viridis")
            ax.set_title(f"{name},  t = {st:.0f} ms")
            ax.set_xlabel(r"$x$ ($\mu$m)")
            ax.set_ylabel(r"$y$ ($\mu$m)")
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Forward problem: PINN vs. analytical and FD references "
                 "(2D snapshots)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(result, savepath):
    """Mid-plane (y = 0) space-time heatmap."""
    xs, ts = result["xs"], result["ts"]
    j_mid = len(result["ys"]) // 2

    C_pinn_xt = result["C_pinn_grid"][:, j_mid, :]   # [nt, nx]
    C_anal_xt = np.stack(
        [C_analytical(xs, 0.0, t) for t in ts], axis=0
    )
    err_xt = np.abs(C_anal_xt - C_pinn_xt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, title in zip(
        axes,
        [C_anal_xt, C_pinn_xt, err_xt],
        ["Analytical (y=0)", "PINN (y=0)", "|Analytical - PINN|"],
    ):
        im = ax.imshow(field, aspect="auto", origin="lower",
                       extent=[-L / 2, L / 2, 0, T], cmap="viridis")
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$t$ (ms)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_inverse_convergence(savepath):
    import re
    fname = FIG_DIR / "variables.dat"
    if not fname.exists():
        return
    iters, Ds, ks = [], [], []
    with open(fname) as fh:
        for line in fh:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            try:
                it = int(parts[0])
            except ValueError:
                continue
            nums = re.findall(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?", parts[1])
            if len(nums) >= 2:
                iters.append(it)
                Ds.append(np.exp(float(nums[0])))
                ks.append(np.exp(float(nums[1])))
    if not iters:
        return
    iters, Ds, ks = np.array(iters), np.array(Ds), np.array(ks)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(iters, Ds, "r-")
    axes[0].axhline(D_TRUE, color="k", ls=":", label="true")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(r"$D$ ($\mu$m$^2$/ms)")
    axes[0].set_title("Diffusion coefficient recovery")
    axes[0].legend()
    axes[1].plot(iters, ks, "b-")
    axes[1].axhline(K_TRUE, color="k", ls=":", label="true")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$k$ (1/ms)")
    axes[1].set_title("Reuptake rate recovery")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# 11. Main
# =============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--skip-inverse", action="store_true")
    parser.add_argument("--noise-pct", type=float, default=2.0,
                        help="Gaussian noise sigma as %% of peak C0 (default 2)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast sanity-check run, not paper-quality")
    args = parser.parse_args()

    if args.quick:
        global ADAM_ITERS, LBFGS_ITERS
        global N_DOMAIN, N_BOUNDARY, N_INITIAL, N_TEST, N_OBS
        ADAM_ITERS  = 1_000
        LBFGS_ITERS = 200
        N_DOMAIN    = 1_000
        N_BOUNDARY  = 80
        N_INITIAL   = 80
        N_TEST      = 500
        N_OBS       = 80
        print("[QUICK MODE] Reduced hyperparameters for fast sanity check.")

    metrics = {
        "dimension":      "2D",
        "implementation": "JAX/Flax NNX + Optax + jaxopt",
        "true_params":    {"D": D_TRUE, "k": K_TRUE},
        "hyperparameters": {
            "layers":      LAYERS,
            "activation":  "tanh",
            "adam_iters":  ADAM_ITERS,
            "lbfgs_iters": LBFGS_ITERS,
            "n_domain":    N_DOMAIN,
            "noise_pct":   args.noise_pct,
        },
    }

    if not args.skip_forward:
        model_fwd = train_forward()
        fwd = evaluate_forward(model_fwd)
        metrics["forward"] = {
            "L2_vs_analytical_pct": float(fwd["L2_vs_analytical_pct"]),
            "L2_vs_fd_pct":         float(fwd["L2_vs_fd_pct"]),
            "per_snapshot":         fwd["per_snapshot"],
        }
        plot_forward(fwd, FIG_DIR / "forward_snapshots.png")
        plot_heatmap(fwd, FIG_DIR / "forward_heatmap.png")
        print(f"\n  L2(PINN vs. FD)         = {fwd['L2_vs_fd_pct']:.3f}%")
        print(f"  L2(PINN vs. analytical) = {fwd['L2_vs_analytical_pct']:.3f}%")
        for tag, ps in fwd["per_snapshot"].items():
            print(f"    {tag}: anal {ps['L2_anal_pct']:.2f}%, "
                  f"FD {ps['L2_fd_pct']:.2f}%")

    if not args.skip_inverse:
        obs_x, obs_y, obs_t, obs_C = make_noisy_observations(
            n_obs=N_OBS, noise_pct=args.noise_pct,
        )
        _, D_rec, k_rec = train_inverse(obs_x, obs_y, obs_t, obs_C)
        rel_D = 100.0 * abs(D_rec - D_TRUE) / D_TRUE
        rel_k = 100.0 * abs(k_rec - K_TRUE) / K_TRUE
        metrics["inverse"] = {
            "D_recovered":     D_rec,
            "k_recovered":     k_rec,
            "D_rel_error_pct": rel_D,
            "k_rel_error_pct": rel_k,
            "n_observations":  int(N_OBS),
            "noise_pct":       args.noise_pct,
        }
        plot_inverse_convergence(FIG_DIR / "inverse_convergence.png")
        print(f"\n  Recovered D = {D_rec:.4f} mu_m^2/ms (true 0.32)   "
              f"|err| = {rel_D:.2f}%")
        print(f"  Recovered k = {k_rec:.4f} 1/ms        (true 0.05)   "
              f"|err| = {rel_k:.2f}%")

    with open(FIG_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n  Metrics saved to {FIG_DIR / 'metrics.json'}")
    print(f"  Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
