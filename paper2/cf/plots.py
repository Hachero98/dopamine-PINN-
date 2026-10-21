"""Figures from saved studies (no retraining needed).

    plot_study('runs/linear/fast/obs4001_net1234')       # three PNGs in that folder
    plot_compare(load_runs('runs', 'linear', 'fast'), 'runs/linear/fast/compare.png')
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from . import mechanisms  # noqa: E402
from .settings import Design  # noqa: E402
from .model import Forward  # noqa: E402
from .effects import EffectGrid  # noqa: E402
from .study import load_study  # noqa: E402

Z = 1.96


def _fields(path, times):
    """Truth, network and PDE-at-PINN effect fields at the given times (rebuilt)."""
    res, net, p_net = load_study(path)
    mech = mechanisms.get(res['mechanism']['name'], truth=res['mechanism']['truth'], init=res['mechanism']['init'])
    design = Design(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in res['design'].items()})
    fwd = Forward(mech, design, nx=res['profile']['fd_nx'])
    grid = EffectGrid(fwd, res['profile']['eval_dt'])
    idx = [int(np.argmin(np.abs(grid.ts - t))) for t in times]
    a = design.heldout
    X, Y = np.meshgrid(grid.x, grid.x, indexing='ij')
    th_p = fwd.theta(res['pinn']['stage4']['params'])
    tau_true = np.asarray(grid.tau(fwd.theta_true, a))[idx]
    tau_pde = np.asarray(grid.tau(th_p, a))[idx]
    tau_net = np.stack([net.predict(p_net, X, Y, grid.ts[i], a) - net.predict(p_net, X, Y, grid.ts[i], 0.0)
                        for i in idx])
    return res, grid, [grid.ts[i] for i in idx], tau_true, tau_net, tau_pde


def plot_study(path):
    path = Path(path)
    T = json.loads((path / 'results.json').read_text())['design']['T']
    res, grid, ts, tt, tn, tp = _fields(path, (0.1 * T, 0.4 * T, 0.9 * T))
    a = res['design']['heldout']
    ext = [grid.x[0], grid.x[-1], grid.x[0], grid.x[-1]]

    # 1. effect fields: truth and network on one range per time; errors in % of the true peak
    fig, ax = plt.subplots(4, 3, figsize=(12.5, 14))
    rel = lambda e, c: 100 * e / np.abs(tt[c]).max()
    emax = max(max(np.abs(rel(tn[c] - tt[c], c)).max(), np.abs(rel(tp[c] - tt[c], c)).max()) for c in range(3))
    for c in range(3):
        lo, hi = min(tt[c].min(), tn[c].min()), max(tt[c].max(), tn[c].max())
        rows = [(tt[c], 'τ truth (µM)', 'viridis', lo, hi), (tn[c], 'τ network (µM)', 'viridis', lo, hi),
                (rel(tn[c] - tt[c], c), 'network − truth (%)', 'RdBu_r', -emax, emax),
                (rel(tp[c] - tt[c], c), 'PDE at PINN λ̂ − truth (%)', 'RdBu_r', -emax, emax)]
        for r, (F, title, cmap, vlo, vhi) in enumerate(rows):
            im = ax[r, c].imshow(F.T, origin='lower', extent=ext, cmap=cmap, vmin=vlo, vmax=vhi)
            ax[r, c].set_title(f'{title}, t = {ts[c]:g} ms', fontsize=10)
            ax[r, c].set_xlabel('x (µm)')
            ax[r, c].set_ylabel('y (µm)')
            plt.colorbar(im, ax=ax[r, c], fraction=0.046)
    fig.suptitle(f"{res['mechanism']['name']}: held-out effect τ under do(A = {a}); errors in % of the true "
                 "peak at that time")
    fig.tight_layout()
    fig.savefig(path / 'effect_fields.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. effect summaries
    cv, dr = res['curves'], res['dose_response']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    t = np.array(cv['ts'])
    m, sd, bd = np.array(cv['ATE_reference']), np.array(cv['ATE_reference_sd']), np.array(cv['ATE_bound_sd'])
    ax[0].fill_between(t, m - Z * sd, m + Z * sd, color='0.85', label='reference ±1.96 SD')
    ax[0].plot(t, np.array(cv['ATE_truth']) + Z * bd, 'k:', lw=0.8, label='truth ±1.96 bound')
    ax[0].plot(t, np.array(cv['ATE_truth']) - Z * bd, 'k:', lw=0.8)
    for k, st, lab in (('truth', 'k-', 'truth'), ('network', 'r--', 'PINN network'), ('pinn_pde', 'b:', 'PDE at PINN λ̂')):
        ax[0].plot(t, cv[f'ATE_{k}'], st, label=lab)
    ax[0].set(xlabel='t (ms)', ylabel='ATE(t) (µM)', title=f'Domain-averaged effect, A = {a}')
    x = np.array(cv['x'])
    for k, st, lab in (('truth', 'k-', 'truth'), ('network', 'r--', 'PINN network'), ('pinn_pde', 'b:', 'PDE at PINN λ̂'),
                       ('reference', 'g-.', 'reference')):
        ax[1].plot(x, cv[f'CE_centre_{k}'], st, label=lab)
    ax[1].set(xlabel='x (µm), y = 0', ylabel='CE (µM ms)', title='Cumulative effect, centre line')
    d = np.array(dr['doses'])
    m, sd = np.array(dr['reference']), np.array(dr['reference_sd'])
    ax[2].fill_between(d, m - Z * sd, m + Z * sd, color='0.85', label='reference ±1.96 SD')
    for k, st, lab in (('truth', 'k-', 'truth'), ('network', 'r--', 'PINN network'), ('pinn_pde', 'b:', 'PDE at PINN λ̂')):
        ax[2].plot(d, dr[k], st, label=lab)
    for at in res['design']['doses']:
        ax[2].axvline(at, color='0.85', lw=0.8, zorder=0)
    ax[2].axvline(a, color='orange', ls='--', lw=1, label='held-out dose')
    ax[2].set(xlabel='dose A', ylabel='GTE (µM)', title='Dose-response of the global effect')
    for axx in ax:
        axx.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path / 'effect_summaries.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. parameters through the stages
    params, truth, init = res['mechanism']['params'], res['mechanism']['truth'], res['mechanism']['init']
    stages = ['stage2', 'stage3', 'stage4']
    fig, ax = plt.subplots(1, len(params), figsize=(4 * len(params), 3.8))
    for axx, q in zip(ax, params):
        axx.plot(range(4), [init[q]] + [res['pinn'][s]['params'][q] for s in stages], 'ro-', label='PINN')
        axx.plot(range(1, 4), [res['reference'][s]['params'][q] for s in stages], 'bs--', label='reference')
        axx.axhline(truth[q], color='k', ls=':', label='truth')
        axx.set_xticks(range(4))
        axx.set_xticklabels(['init'] + stages)
        axx.set_title(q)
        axx.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path / 'parameters.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_compare(rows, path):
    """PINN vs reference errors on the same records, with the bound."""
    if not rows:
        return
    params = rows[0]['mechanism']['params']
    items = [(q, lambda r, q=q: r['pinn']['stage4']['err_log_pct'][q], lambda r, q=q: r['reference']['stage4']['err_log_pct'][q],
              lambda r, q=q: r['bound']['sd_log_pct'][q]) for q in params]
    items.append(('GTE', lambda r: r['heldout']['pinn_pde']['GTE_err_pct'],
                  lambda r: r['heldout']['reference']['GTE_err_pct'], lambda r: r['bound']['GTE_sd_pct']))
    fig, ax = plt.subplots(1, len(items) + 1, figsize=(4.2 * (len(items) + 1), 4.2))
    for axx, (q, fp, fr, fb) in zip(ax, items):
        p, r = np.array([fp(x) for x in rows]), np.array([fr(x) for x in rows])
        c = Z * np.sqrt(np.mean(np.array([fb(x) for x in rows]) ** 2))
        lim = 1.1 * max(np.abs(p).max(), np.abs(r).max(), c)
        axx.axvspan(-c, c, color='0.92', zorder=0, label='±1.96 bound')
        axx.axhspan(-c, c, color='0.92', zorder=0)
        axx.plot([-lim, lim], [-lim, lim], 'k-', lw=0.6)
        axx.scatter(r, p, s=16, c='C3')
        axx.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='reference error (%)', ylabel='PINN error (%)',
                title=f'{q} ({"log, " if q != "GTE" else ""}%)')
        axx.legend(fontsize=8, loc='upper left')
    axx = ax[-1]
    xs = np.arange(len(rows))
    axx.bar(xs - 0.2, [r['heldout']['reference']['tau_L2_pct'] for r in rows], 0.4, label='reference')
    axx.bar(xs + 0.2, [r['heldout']['pinn_pde']['tau_L2_pct'] for r in rows], 0.4, color='C3', label='PINN (PDE at λ̂)')
    axx.axhline(np.sqrt(np.mean([r['bound']['tau_L2_rmse_pct'] ** 2 for r in rows])), color='k', ls='--', lw=0.8,
                label='RMS bound')
    axx.set_xticks(xs)
    axx.set_xticklabels([str(r['seeds']['obs']) for r in rows], rotation=90, fontsize=7)
    axx.set(ylabel='L2 error of τ (%)', title='Held-out effect field')
    axx.legend(fontsize=8)
    fig.suptitle(f"{rows[0]['mechanism']['name']} | {rows[0]['profile']['name']} | {len(rows)} realizations")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
