"""Finite-difference reference solver for dC/dt = div(D grad C) - s(C) on a square.

Node-centred flux form on an nx x nx grid over [-L/2, L/2]^2:

    dC_ij/dt = [F_{i+1/2,j} - F_{i-1/2,j}] / (w_i dx^2) + [...]^y - s(C_ij),
    F_{i+1/2,j} = Dbar (C_{i+1,j} - C_ij),

with Dbar the mean of the two nodal values and w_i = 1/2 on the walls (half control
volume, zero wall flux). For constant D this is the mirror-Neumann explicit scheme of
the original notebooks (`fd_reference_np`). An optional obstacle mask sets the flux
through every face that touches a solid node to zero. The time step is fixed by a
static bound `D_cap`, so the solver is differentiable in D and the uptake parameters
(forward mode), which gives the reference estimator exact Jacobians.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import config as cfg


def grid(nx, L=cfg.L):
    return np.linspace(-L / 2, L / 2, nx)


def gaussian_ic(nx, amp=cfg.C0, sigma=cfg.SIGMA, L=cfg.L):
    x = grid(nx, L)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return amp * np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def make_fd_solver(nx=cfg.NX_FD, D_cap=cfg.D_CAP, mask=None, T_end=cfg.T, dt_store=cfg.DT_STORE, L=cfg.L):
    """Return solve(D_nodes, sink, C_init) -> frames of shape (n_frames + 1, nx, nx).

    Frames are stored at t = j * dt_store, j = 0 .. T_end / dt_store. D_nodes is a
    scalar or an (nx, nx) array with D <= D_cap; sink(C) is the uptake rate, e.g.
    `lambda C: k * C`. Attributes nx, dx, dt, dt_store, T_end and mask describe the run.
    """
    dx = L / (nx - 1)
    dt_max = 0.95 * dx**2 / (4.0 * D_cap)
    stride = int(np.ceil(dt_store / dt_max))
    dt = dt_store / stride
    n_frames = int(round(T_end / dt_store))
    m = np.ones((nx, nx)) if mask is None else mask.astype(float)
    mfx = jnp.asarray(m[1:, :] * m[:-1, :])
    mfy = jnp.asarray(m[:, 1:] * m[:, :-1])
    w = np.ones(nx)
    w[0] = w[-1] = 0.5
    ivx = jnp.asarray(1.0 / (dx**2 * w))[:, None]
    ivy = jnp.asarray(1.0 / (dx**2 * w))[None, :]
    mj = jnp.asarray(m)

    def solve(D_nodes, sink, C_init):
        Dn = jnp.broadcast_to(jnp.asarray(D_nodes, jnp.float64), (nx, nx))
        Dfx = 0.5 * (Dn[1:, :] + Dn[:-1, :]) * mfx
        Dfy = 0.5 * (Dn[:, 1:] + Dn[:, :-1]) * mfy

        def step(_, C):
            Fx = Dfx * (C[1:, :] - C[:-1, :])
            Fy = Dfy * (C[:, 1:] - C[:, :-1])
            div = ((jnp.pad(Fx, ((0, 1), (0, 0))) - jnp.pad(Fx, ((1, 0), (0, 0)))) * ivx
                   + (jnp.pad(Fy, ((0, 0), (0, 1))) - jnp.pad(Fy, ((0, 0), (1, 0)))) * ivy)
            return C + dt * (div - sink(C)) * mj

        def frame(C, _):
            C = lax.fori_loop(0, stride, step, C)
            return C, C

        C_init = jnp.asarray(C_init) * mj
        _, F = lax.scan(frame, C_init, None, length=n_frames)
        return jnp.concatenate([C_init[None], F])

    solve.nx, solve.dx, solve.dt, solve.stride = nx, dx, dt, stride
    solve.dt_store, solve.T_end, solve.mask, solve.L, solve.D_cap = dt_store, T_end, m, L, D_cap
    return solve


def interp_frames(frames, dx, xq, yq, tq, dt_store=cfg.DT_STORE, L=cfg.L):
    """Trilinear interpolation of stored frames at points (x, y, t); differentiable."""
    nf, nx, _ = frames.shape
    ft = tq / dt_store
    it = jnp.clip(jnp.floor(ft).astype(int), 0, nf - 2)
    a = ft - it
    fx = (xq + L / 2) / dx
    ix = jnp.clip(jnp.floor(fx).astype(int), 0, nx - 2)
    b = fx - ix
    fy = (yq + L / 2) / dx
    iy = jnp.clip(jnp.floor(fy).astype(int), 0, nx - 2)
    c = fy - iy

    def bil(ti):
        return ((1 - b) * (1 - c) * frames[ti, ix, iy] + b * (1 - c) * frames[ti, ix + 1, iy]
                + (1 - b) * c * frames[ti, ix, iy + 1] + b * c * frames[ti, ix + 1, iy + 1])
    return (1 - a) * bil(it) + a * bil(it + 1)


def solve_at(solver, D, sink, amp=cfg.C0):
    """Convenience: frames for a Gaussian release of peak `amp`."""
    return solver(D, sink, gaussian_ic(solver.nx, amp, L=solver.L))


def linear_sink(k):
    return lambda C: k * C


def mm_sink(Vmax, Km):
    return lambda C: Vmax * C / (Km + C)


def total_mass(frames, solver):
    """Discrete total amount: sum of C over control volumes (half volumes on the walls)."""
    w = np.ones(solver.nx)
    w[0] = w[-1] = 0.5
    W = np.outer(w, w) * solver.dx**2 * solver.mask
    return jnp.sum(frames * W, axis=(-2, -1))


def analytical_infinite(x, y, t, D=cfg.D_TRUE, k=cfg.K_TRUE, sigma=cfg.SIGMA, C0=cfg.C0):
    """Closed-form solution on the infinite plane (valid here only before the walls matter)."""
    s2 = sigma**2 + 2.0 * D * t
    return C0 * sigma**2 / s2 * np.exp(-(x**2 + y**2) / (2.0 * s2)) * np.exp(-k * t)


def fd_reference_np(D=cfg.D_TRUE, k=cfg.K_TRUE, nx=81, T_end=cfg.T, dt_store=0.02, L=cfg.L,
                    sigma=cfg.SIGMA, C0=cfg.C0):
    """The NumPy explicit solver of the original notebooks (paper v1-v5 main pipeline).

    Kept for cross-checks: returns (x, t_frames, frames). Same mirror-Neumann scheme as
    `make_fd_solver` with constant D; only the time step differs.
    """
    dx = L / (nx - 1)
    nt = int(np.ceil(T_end / (0.95 / (2.0 * D * (2.0 / dx**2))))) + 1
    dt = T_end / (nt - 1)
    stride = max(1, int(round(dt_store / dt)))
    C = gaussian_ic(nx, C0, sigma, L)
    frames, ts = [C.copy()], [0.0]
    for n in range(1, nt):
        lap = np.zeros_like(C)
        lap[1:-1, :] += (C[2:, :] - 2 * C[1:-1, :] + C[:-2, :]) / dx**2
        lap[0, :] += 2 * (C[1, :] - C[0, :]) / dx**2
        lap[-1, :] += 2 * (C[-2, :] - C[-1, :]) / dx**2
        lap[:, 1:-1] += (C[:, 2:] - 2 * C[:, 1:-1] + C[:, :-2]) / dx**2
        lap[:, 0] += 2 * (C[:, 1] - C[:, 0]) / dx**2
        lap[:, -1] += 2 * (C[:, -2] - C[:, -1]) / dx**2
        C = C + D * dt * lap - k * dt * C
        if n % stride == 0 or n == nt - 1:
            frames.append(C.copy())
            ts.append(n * dt)
    return grid(nx, L), np.array(ts), np.stack(frames)


def disc_mask(nx, discs, L=cfg.L):
    """Boolean mask of extracellular nodes for impermeable discs [(xc, yc, R), ...]."""
    x = grid(nx, L)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.all([(X - a)**2 + (Y - b)**2 > R**2 for a, b, R in discs], axis=0)
