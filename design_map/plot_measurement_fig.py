"""Measurement-model figure (BSPC manuscript, Fig. 4): realistic electrode
effects, naive vs measurement-aware fits.

a  site r = 0.5 um: true point concentration, noise-free sensor output and one
   measured record, all effects combined (M5)
b  D: median signed error of the reference estimator, naive and aware, per
   scenario; PINN aware fits as diamonds when present
c  same for k
d  coverage of the reference 95% intervals with exact binomial 95% CIs

Usage:  python plot_measurement_fig.py <measurement_model.json> <manuscript_dir> <figures_dir> [fig_no]
Writes Fig<n>.pdf (manuscript) and Figure_<n>.{eps,tif} (upload) at 174 mm width.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

meas = json.loads(Path(sys.argv[1]).read_text())
MS_DIR, FIG_DIR = Path(sys.argv[2]), Path(sys.argv[3])
NO = sys.argv[4] if len(sys.argv) > 4 else '4'

RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
      'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
      'legend.fontsize': 6.5, 'lines.linewidth': 1.2, 'lines.markersize': 4}
GRID = dict(color='0.88', lw=0.5)
W = 174 / 25.4
LAB = {'M0_ideal': 'ideal', 'M1_tip': 'tip\naveraging', 'M2_lag': 'response\nlag',
       'M3_colour': 'correlated\nnoise', 'M4_drift': 'baseline\noffset', 'M5_all': 'all\ncombined'}
C_NAIVE, C_AWARE = '#c0392b', '#1e8449'

names = [n for n in LAB if n in meas['scenarios']]
S = meas['scenarios']
xi = np.arange(len(names))


def med(n, mode, key):
    return float(np.median([d[mode][f'{key}_err'] for d in S[n]['ref'].values()]))


def ref5_med(n, key):
    """Median of the aware reference fit on the realizations fitted by the PINN."""
    seeds = sorted({str(p['noise_seed']) for p in S[n].get('pinn', {}).values()})
    return float(np.median([S[n]['ref'][s_]['aware'][f'{key}_err'] for s_ in seeds])) if seeds else None


def pinn_med(n, mode, key):
    v = [p[f'{key}_err'] for p in S[n].get('pinn', {}).values() if p['mode'] == mode]
    return float(np.median(v)) if v else None


with plt.rc_context(RC):
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.9))

    # (a) signal at the near site, all effects combined
    ax = axes[0, 0]
    ser = S['M5_all']['series'] if 'M5_all' in S else S[names[-1]]['series']
    t = np.array(ser['t'])
    n_t = t.size
    ax.plot(t, np.array(ser['noisy'])[:n_t], '.', ms=2.2, color='0.6', label='measured record')
    ax.plot(t, np.array(ser['point'])[:n_t], '-', color='k', lw=1.0, label='true concentration at the site')
    ax.plot(t, np.array(ser['clean'])[:n_t], '--', color=C_NAIVE, lw=1.1, label='sensor output without noise')
    ax.axvspan(t[0], 0, color='0.93', lw=0)
    ax.text(t[0] / 2, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.3, 'baseline',
            ha='center', va='top', fontsize=6.5, color='0.35')
    ax.set_xlabel('time after release (ms)')
    ax.set_ylabel(r'concentration ($\mu$M)')
    ax.grid(**GRID); ax.legend(loc='upper right', frameon=False)
    ax.set_title('a', loc='left', fontweight='bold')

    # (b, c) median signed error per scenario
    for ax, key, sym, tag in ((axes[0, 1], 'D', '$D$', 'b'), (axes[1, 0], 'k', '$k$', 'c')):
        nv = [med(n, 'naive', key) for n in names]
        av = [med(n, 'aware', key) for n in names]
        ax.axhline(0, color='k', lw=0.7)
        ax.plot(xi - 0.12, nv, 'o', mfc='white', mec=C_NAIVE, mew=1.2, label='reference, naive fit')
        ax.plot(xi + 0.12, av, 'o', color=C_AWARE, label='reference, measurement-aware fit')
        has_p = False
        for i, n in enumerate(names):
            for mode, dx, col in (('aware', 0.12, C_AWARE), ('naive', -0.12, C_NAIVE)):
                pv = pinn_med(n, mode, key)
                if pv is not None:
                    ax.plot(xi[i] + dx + 0.12, pv, 'd', mfc='white', mec=col, mew=1.0, ms=4.5,
                            label=None)
                    has_p = True
        if has_p:
            r5 = [ref5_med(n, key) for n in names]
            ax.plot(xi + 0.38, r5, 'x', color=C_AWARE, mew=1.0, ms=4.5, zorder=4)
            ax.plot([], [], 'd', mfc='white', mec='0.3', mew=1.0, ms=4.5, label='PINN (5 realizations)')
            ax.plot([], [], 'x', color=C_AWARE, mew=1.0, ms=4.5, label='reference, aware, same 5')
        ax.set_yscale('symlog', linthresh=5.0, linscale=1.0)
        ticks = [-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 200]
        pv_all = [v for n in names for mode in ('aware', 'naive')
                  for v in [pinn_med(n, mode, key)] if v is not None]
        pv_all += [v for n in names for v in [ref5_med(n, key)] if v is not None]
        allv = nv + av + pv_all
        lo_, hi_ = min(allv) * 1.4 - 2, max(allv) * 1.4 + 2
        ax.set_yticks([v for v in ticks if lo_ <= v <= hi_])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:+g}' if v else '0'))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(lo_, hi_)
        ax.set_xticks(xi); ax.set_xticklabels([LAB[n] for n in names], fontsize=6.3)
        ax.set_ylabel(f'median error of {sym} (%)')
        ax.grid(axis='y', **GRID)
        if key == 'k':
            fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=4,
                       frameon=False, bbox_to_anchor=(0.5, 0.0))
        ax.set_title(tag, loc='left', fontweight='bold')

    # (d) coverage of the reference 95% intervals
    ax = axes[1, 1]
    for j, (mode, key, col, mk) in enumerate((('naive', 'D', C_NAIVE, 'o'), ('aware', 'D', C_AWARE, 'o'),
                                             ('naive', 'k', C_NAIVE, 's'), ('aware', 'k', C_AWARE, 's'))):
        sm = [S[n]['summary'][mode] for n in names]
        cov = np.array([100 * s[f'ref_cover_{key}'] / s['n_ref'] for s in sm])
        lo = cov - np.array([s[f'ref_cover_{key}_ci'][0] for s in sm])
        hi = np.array([s[f'ref_cover_{key}_ci'][1] for s in sm]) - cov
        ax.errorbar(xi + (j - 1.5) * 0.14, cov, yerr=[lo, hi], fmt=mk, color=col,
                    mfc='white' if mode == 'naive' else col, mec=col, ms=3.5, capsize=1.5, lw=0.8,
                    label=f'{mode}, {key}')
    ax.axhline(95, color='k', ls=':', lw=0.9)
    ax.set_ylim(-4, 104)
    ax.set_xticks(xi); ax.set_xticklabels([LAB[n] for n in names], fontsize=6.3)
    ax.set_ylabel('coverage of 95% intervals (%)')
    ax.grid(axis='y', **GRID); ax.legend(loc='lower left', frameon=False, ncol=2)
    ax.set_title('d', loc='left', fontweight='bold')

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(MS_DIR / f'Fig{NO}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'Figure_{NO}.eps', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'Figure_{NO}.tif', bbox_inches='tight', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(Path(sys.argv[1]).with_name('measurement_fig_preview.png'), bbox_inches='tight', dpi=200)
print('done')
