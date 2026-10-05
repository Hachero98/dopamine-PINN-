"""Frozen benchmark datasets: generation, saving and loading.

Every observation set used in the baseline paper (and the extension experiments) is
regenerated here from its seed with the package's FD solver and stored as an .npz file
plus an entry in index.json (truth, design, solver settings, SHA-256). Later studies load
these files, so they compare on exactly the same data.

    from dopamine_pinn import benchmark
    obs, meta = benchmark.load('linear_s4001')
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import config as cfg
from . import __version__
from .solver import make_fd_solver, solve_at, linear_sink, mm_sink, disc_mask, gaussian_ic
from . import observations as ob

DEFAULT_DIR = Path(__file__).resolve().parent.parent / 'benchmark' / 'v1'

OBSTACLES = [(1.5, 0.2, 0.55), (-1.4, 0.9, 0.5), (0.3, 1.6, 0.5), (-0.6, -1.6, 0.55),
             (1.2, -1.5, 0.45), (-1.7, -0.6, 0.45), (1.8, 1.7, 0.4), (-1.8, 1.9, 0.35)]
DFIELD = dict(A=0.5, xc=0.8, yc=0.4, w=0.6)


def d_field_true(X, Y, D0=cfg.D_TRUE, A=DFIELD['A'], xc=DFIELD['xc'], yc=DFIELD['yc'], w=DFIELD['w']):
    return D0 * (1 - A * np.exp(-((X - xc)**2 + (Y - yc)**2) / (2 * w**2)))


def _solver_meta(s):
    return {'nx': s.nx, 'dx': s.dx, 'dt': s.dt, 'dt_store': s.dt_store, 'T_end': s.T_end, 'D_cap': s.D_cap}


def specs():
    """All benchmark datasets: name -> (builder, metadata). Builders return obs dicts."""
    S = {}
    lin = make_fd_solver()
    cache = {}

    def frames(D=cfg.D_TRUE, k=cfg.K_TRUE):
        if (D, k) not in cache:
            cache[(D, k)] = solve_at(lin, D, linear_sink(k))
        return cache[(D, k)]

    base = {'model': 'linear', 'D': cfg.D_TRUE, 'k': cfg.K_TRUE, 'C0': cfg.C0, 'sigma': cfg.SIGMA, 'L': cfg.L,
            'solver': _solver_meta(lin)}
    seeds = sorted({s for v in cfg.SEEDS.values() for s in v})
    for s in seeds:
        S[f'linear_s{s}'] = (lambda s=s: ob.scattered(frames(), lin, s),
                             {**base, 'design': 'scattered', 'n_obs': cfg.N_OBS, 'noise_sd': cfg.NOISE_SD,
                              't_window': cfg.T_WINDOW, 'seed': s})
    for eta in (0.0, 1.0, 2.0, 5.0, 10.0):
        S[f'linear_noise{eta:g}_s4001'] = (
            lambda eta=eta: ob.scattered(frames(), lin, 4001, noise_sd=eta / 100 * cfg.C0),
            {**base, 'design': 'scattered', 'n_obs': cfg.N_OBS, 'noise_sd': eta / 100, 't_window': cfg.T_WINDOW,
             'seed': 4001, 'note': 'same locations and standardized noise at every level'})
    for D, k in ((0.16, 0.010), (0.16, 0.040), (0.32, 0.020), (0.50, 0.010), (0.50, 0.040)):
        S[f'linear_grid_D{D:g}_k{k:g}_s4001'] = (
            lambda D=D, k=k: ob.scattered(frames(D, k), lin, 4001),
            {**base, 'D': D, 'k': k, 'design': 'scattered', 'n_obs': cfg.N_OBS, 'noise_sd': cfg.NOISE_SD,
             't_window': cfg.T_WINDOW, 'seed': 4001})
    for n in (100, 200, 400, 800, 1600):
        S[f'linear_n{n}_s4001'] = (lambda n=n: ob.scattered(frames(), lin, 4001, n_obs=n),
                                   {**base, 'design': 'scattered', 'n_obs': n, 'noise_sd': cfg.NOISE_SD,
                                    't_window': cfg.T_WINDOW, 'seed': 4001})
    for name, d in ob.ELECTRODE_DESIGNS.items():
        S[f'electrode_{name}_s4001'] = (lambda d=d: ob.electrode(frames(), lin, d['sites'], d['dt'], 4001),
                                        {**base, 'design': 'electrode', 'sites': d['sites'], 'dt': d['dt'],
                                         'noise_sd': cfg.NOISE_SD, 't_window': cfg.T_WINDOW, 'seed': 4001})

    # Michaelis-Menten uptake (extension E3): 50 ms at 1 uM and 500 ms at 10 uM
    for name, T_end, amp, dts in (('mm_T50_C1_s9001', 50.0, 1.0, cfg.DT_STORE),
                                  ('mm_T500_C10_s9001', 500.0, 10.0, 0.5)):
        def build(T_end=T_end, amp=amp, dts=dts):
            s = make_fd_solver(T_end=T_end, dt_store=dts)
            fr = s(cfg.D_TRUE, mm_sink(cfg.VMAX_TRUE, cfg.KM_TRUE), gaussian_ic(s.nx, amp))
            return ob.scattered(fr, s, 9001, t_window=(0.5, 0.9 * T_end))
        S[name] = (build, {'model': 'michaelis-menten', 'D': cfg.D_TRUE, 'Vmax': cfg.VMAX_TRUE, 'Km': cfg.KM_TRUE,
                           'C0': amp, 'sigma': cfg.SIGMA, 'L': cfg.L, 'design': 'scattered', 'n_obs': cfg.N_OBS,
                           'noise_sd': cfg.NOISE_SD, 't_window': (0.5, 0.9 * T_end), 'seed': 9001,
                           'solver': _solver_meta(make_fd_solver(T_end=T_end, dt_store=dts))})

    # spatially varying D (extension E4): 600 early + 400 full-window points
    def build_dfield():
        x = np.linspace(-cfg.L / 2, cfg.L / 2, lin.nx)
        X, Y = np.meshgrid(x, x, indexing='ij')
        fr = lin(d_field_true(X, Y), linear_sink(cfg.K_TRUE), gaussian_ic(lin.nx))
        return ob.merge(ob.scattered(fr, lin, 9101, n_obs=600, t_window=(0.2, 10.0)),
                        ob.scattered(fr, lin, 9102))
    S['dfield_s9101'] = (build_dfield, {**base, 'model': 'linear, D(x, y)', 'D_field': DFIELD,
                                        'design': '600 scattered in 0.2-10 ms + 400 in 0.5-45 ms',
                                        'noise_sd': cfg.NOISE_SD, 'seed': [9101, 9102]})

    # obstacles (extension E5): free D in the gaps, 161-node grid, node observations
    def build_obstacles():
        s = make_fd_solver(161, D_cap=1.25, mask=disc_mask(161, OBSTACLES))
        fr = s(cfg.D_FREE, linear_sink(cfg.K_TRUE), gaussian_ic(161))
        return ob.grid_nodes(fr, s, 9201)
    S['obstacles_s9201'] = (build_obstacles, {
        'model': 'linear, obstacles', 'D_free': cfg.D_FREE, 'k': cfg.K_TRUE, 'C0': cfg.C0, 'sigma': cfg.SIGMA,
        'L': cfg.L, 'obstacles': OBSTACLES, 'design': 'extracellular grid nodes', 'n_obs': cfg.N_OBS,
        'noise_sd': cfg.NOISE_SD, 't_window': cfg.T_WINDOW, 'seed': 9201,
        'solver': _solver_meta(make_fd_solver(161, D_cap=1.25))})
    return S


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(out_dir=DEFAULT_DIR, names=None, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for name, (build, meta) in specs().items():
        if names is not None and name not in names:
            continue
        obs = build()
        path = out_dir / f'{name}.npz'
        np.savez(path, **{k: np.asarray(v, np.float64) for k, v in obs.items()})
        index[name] = {**json.loads(json.dumps(meta)), 'n': int(len(obs['C_d'])), 'file': path.name,
                       'sha256': _sha(path), 'generator': f'dopamine_pinn {__version__}'}
        if verbose:
            print(f'  {name:<34} n = {len(obs["C_d"]):>4}')
    index_path.write_text(json.dumps(index, indent=1))
    return index


def load(name, data_dir=DEFAULT_DIR, check=True):
    """Return (obs, meta) for a benchmark dataset; verifies the SHA-256 by default."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / 'index.json').read_text())[name]
    path = data_dir / meta['file']
    if check and _sha(path) != meta['sha256']:
        raise ValueError(f'{path} does not match its recorded SHA-256')
    with np.load(path) as z:
        return {k: z[k] for k in z.files}, meta


def names(data_dir=DEFAULT_DIR):
    return list(json.loads((Path(data_dir) / 'index.json').read_text()))
