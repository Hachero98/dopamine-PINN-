"""Print-spec figure of the experimental-design map (reads design_map.json).

a  CRB relative SD of D vs electrode distance, one curve per sampling interval
b  same for k
c  CRB vs actual reference-estimator spread over noise realizations (validation)

Writes design_map_fig.{eps,tif,png} at 174 mm width, 8-pt sans-serif.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
m = json.loads((HERE / 'design_map.json').read_text())
RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
      'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
      'legend.fontsize': 6.5, 'lines.linewidth': 1.2, 'lines.markersize': 4}
GRID = dict(color='0.88', lw=0.5)
W = 174 / 25.4

A = m['single_site_axis']
r = np.array(A['r_um'])
cmap = plt.get_cmap('viridis')
with plt.rc_context(RC):
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.35))
    for ax, key, lab in ((axes[0], 'sd_D', r'best achievable SD of $D$ (%)'),
                         (axes[1], 'sd_k', r'best achievable SD of $k$ (%)')):
        for i, dt in enumerate(A['dt_ms']):
            ax.semilogy(r, A[key][i], '-', color=cmap(i / (len(A['dt_ms']) - 1)),
                        label=f'{dt:g} ms')
        ax.axhline(m['scattered_400'][key], color='k', ls='--', lw=0.9,
                   label='400 scattered points')
        ax.set_xlabel(r'electrode distance from release site ($\mu$m)')
        ax.set_ylabel(lab); ax.grid(which='major', **GRID)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, title='sampling interval', title_fontsize=6.5, frameon=False,
               ncol=len(l), loc='lower center', bbox_to_anchor=(0.36, -0.02))
    # validation
    mc = m['monte_carlo']
    names = [n for n in mc if n != 'n_draws']
    xi = np.arange(len(names))
    for j, (key, col) in enumerate((('D', 'C3'), ('k', 'C0'))):
        crb = [mc[n][f'crb_sd_{key}'] for n in names]
        emp = [mc[n][f'empirical_sd_{key}'] for n in names]
        off = (j - 0.5) * 0.3
        axes[2].bar(xi + off, crb, 0.3, color=col, alpha=0.35, label=f'bound, {key}')
        axes[2].plot(xi + off, emp, 'o', color=col, mec='k', mew=0.4,
                     label=f'reference estimator, {key}')
    axes[2].set_xticks(xi)
    axes[2].set_xticklabels([n.replace(' dt=0.5', '').replace('S1 ', '1 site, ').replace('S3', '3 sites')
                             for n in names], fontsize=6.5)
    axes[2].set_ylabel('SD over noise realizations (%)')
    axes[2].grid(axis='y', **GRID); axes[2].legend(frameon=False, loc='upper left', ncol=2)
    axes[2].set_ylim(0, 1.45 * axes[2].get_ylim()[1])
    for ax, ch in zip(axes, 'abc'):
        ax.annotate(ch, xy=(0, 1), xycoords='axes fraction', xytext=(-30, 4),
                    textcoords='offset points', fontsize=9, fontweight='bold')
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for ext, kw in (('png', dict(dpi=300)), ('eps', {}),
                    ('tif', dict(dpi=600, pil_kwargs={'compression': 'tiff_lzw'}))):
        fig.savefig(HERE / f'design_map_fig.{ext}', bbox_inches='tight', **kw)
print('wrote design_map_fig.{png,eps,tif}')
