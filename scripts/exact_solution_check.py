"""Table S2.1 of Online Resource 2: FD reference and whole-plane solution against the
exact solution of the bounded problem.

The bounded problem (main text, equations 1-4) is
    dC/dt = D lap(C) - k C on the square [-L/2, L/2]^2, zero flux on the walls,
    C(x, y, 0) = C0 exp(-(x^2 + y^2) / (2 sigma^2)).
For a centred release on a square with zero-flux walls, its exact solution follows from
the method of images: mirror images of the release sit at every point (mL, nL),
    C_Omega = C0 sigma^2 / (sigma^2 + 2Dt) * sum_{m,n} exp(-((x-mL)^2 + (y-nL)^2) / (2(sigma^2 + 2Dt))) * exp(-kt).
The image terms change the initial condition inside the square by at most
C0 exp(-L^2 / (8 sigma^2)), about 3.7e-6 C0 here.

The script compares, on the 41 x 41 evaluation grid of the main text,
  * the explicit finite-difference reference of dopamine_PINN.ipynb (81 x 81 nodes,
    mirror-Neumann walls, same time step), and
  * the closed-form whole-plane solution (Online Resource 2, equation 8)
with the exact bounded solution, and writes results/exact_solution_check.json.
Deterministic, NumPy only, about a minute on a laptop.

    python scripts/exact_solution_check.py
"""
import json
from pathlib import Path

import numpy as np

D, K, L, SIGMA, C0, T = 0.32, 0.020, 5.0, 0.5, 1.0, 50.0
N_IMAGES = 10          # images per direction; N = 10 and N = 14 agree to machine precision


def fd_reference(nx=81, dt_store=0.02):
    """The explicit FD scheme of dopamine_PINN.ipynb (mirror-Neumann walls)."""
    dx = L / (nx - 1)
    nt = int(np.ceil(T / (0.95 / (2.0 * D * (2.0 / dx**2))))) + 1
    dt = T / (nt - 1)
    stride = max(1, int(round(dt_store / dt)))
    x = np.linspace(-L / 2, L / 2, nx)
    X, Y = np.meshgrid(x, x, indexing='ij')
    C = C0 * np.exp(-(X**2 + Y**2) / (2 * SIGMA**2))
    frames, times = [C.copy()], [0.0]
    for n in range(1, nt):
        lap = np.zeros_like(C)
        lap[1:-1, :] += (C[2:, :] - 2 * C[1:-1, :] + C[:-2, :]) / dx**2
        lap[0, :] += 2 * (C[1, :] - C[0, :]) / dx**2
        lap[-1, :] += 2 * (C[-2, :] - C[-1, :]) / dx**2
        lap[:, 1:-1] += (C[:, 2:] - 2 * C[:, 1:-1] + C[:, :-2]) / dx**2
        lap[:, 0] += 2 * (C[:, 1] - C[:, 0]) / dx**2
        lap[:, -1] += 2 * (C[:, -2] - C[:, -1]) / dx**2
        C = C + D * dt * lap - K * dt * C
        if n % stride == 0 or n == nt - 1:
            frames.append(C.copy())
            times.append(n * dt)
    return x, np.array(times), np.array(frames)


def images_1d(x, t, n_images=N_IMAGES):
    s2 = SIGMA**2 + 2 * D * t
    return sum(np.sqrt(SIGMA**2 / s2) * np.exp(-(x - n * L)**2 / (2 * s2))
               for n in range(-n_images, n_images + 1))


def exact_bounded(X, Y, t):
    return C0 * images_1d(X, t) * images_1d(Y, t) * np.exp(-K * t)


def whole_plane(X, Y, t):
    s2 = SIGMA**2 + 2 * D * t
    return C0 * SIGMA**2 / s2 * np.exp(-(X**2 + Y**2) / (2 * s2)) * np.exp(-K * t)


def rel_l2(a, b):
    return float(100 * np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    x, ts, frames = fd_reference()
    xs = x[::2]                                    # 41 x 41 evaluation grid of the main text
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    out = {'parameters': {'D': D, 'k': K, 'L': L, 'sigma': SIGMA, 'C0': C0, 'images_per_direction': N_IMAGES},
           'max_initial_image_contribution_over_C0': float(np.exp(-L**2 / (8 * SIGMA**2))),
           'snapshots': {}}
    print(f"{'t (ms)':>7} | {'FD vs exact':>12} | {'whole plane vs exact':>21}")
    for t in (0.5, 1, 5, 10, 20, 50):
        i = int(np.argmin(np.abs(ts - t)))
        E = exact_bounded(X, Y, ts[i])
        fd = rel_l2(frames[i][::2, ::2], E)
        wp = rel_l2(whole_plane(X, Y, ts[i]), E)
        out['snapshots'][f'{t:g}'] = {'fd_vs_exact_pct': fd, 'whole_plane_vs_exact_pct': wp}
        print(f'{t:>7g} | {fd:>11.3f}% | {wp:>20.2f}%')
    grid = np.linspace(0, T, 51)
    idx = [int(np.argmin(np.abs(ts - t))) for t in grid]
    E = np.array([exact_bounded(X, Y, ts[i]) for i in idx])
    F = np.array([frames[i][::2, ::2] for i in idx])
    W = np.array([whole_plane(X, Y, ts[i]) for i in idx])
    out['space_time_grid'] = {'fd_vs_exact_pct': rel_l2(F, E), 'whole_plane_vs_exact_pct': rel_l2(W, E)}
    print(f"{'all':>7} | {out['space_time_grid']['fd_vs_exact_pct']:>11.3f}% | "
          f"{out['space_time_grid']['whole_plane_vs_exact_pct']:>20.1f}%")
    path = Path(__file__).resolve().parent.parent / 'results' / 'exact_solution_check.json'
    path.write_text(json.dumps(out, indent=1))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
