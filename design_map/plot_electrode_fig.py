"""Information content of electrode-like designs (BSPC manuscript, Fig. 3).

a  Cramer-Rao bound (CRB) on the relative SD of D vs electrode distance,
   one curve per sampling interval (design_map.json)
b  same for k
c  D: bound, reference-estimator RMSE over 50 realizations (review_analyses.json,
   when given) and over the five realizations fitted by the PINN, and PINN RMSE
   (electrode_design.json)
d  same for k

Writes Fig<n>.pdf (manuscript) and Figure_<n>.{eps,tif} (upload) at 174 mm width.
Usage:  python plot_electrode_fig.py <manuscript_dir> <figures_dir> [fig_no] [review_analyses.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
m = json.loads((HERE / 'design_map.json').read_text())
el = json.loads((HERE / 'electrode_design.json').read_text())
MS_DIR, FIG_DIR = Path(sys.argv[1]), Path(sys.argv[2])
NO = sys.argv[3] if len(sys.argv) > 3 else '7'
R50 = json.loads(Path(sys.argv[4]).read_text())['electrode'] if len(sys.argv) > 4 else None

RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
      'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
      'legend.fontsize': 6.5, 'lines.linewidth': 1.2, 'lines.markersize': 4}
GRID = dict(color='0.88', lw=0.5)
W = 174 / 25.4

A = m['single_site_axis']
r = np.array(A['r_um'])
i_r1 = A['r_um'].index(1.0)
i_dt5 = A['dt_ms'].index(5.0)
i_dt05 = A['dt_ms'].index(0.5)
ms = m['multi_site_dt0.5']
crb = {'S1_r1': ms['1 site (r=1)'],
       'S1_r2': {'sd_D': A['sd_D'][i_dt05][A['r_um'].index(2.0)],
                 'sd_k': A['sd_k'][i_dt05][A['r_um'].index(2.0)]},
       'S2': ms['2 sites (r=0.5, 2)'], 'S3': ms['3 sites (r=1, 2, 2.5)'],
       'S4': ms['4 sites (r=0.5, 1, 1.5, 2)'],
       'S1_r1_coarse': {'sd_D': A['sd_D'][i_dt5][i_r1], 'sd_k': A['sd_k'][i_dt5][i_r1]}}
labels = {'S1_r1': '1 site\n1 µm', 'S1_r2': '1 site\n2 µm', 'S2': '2 sites',
          'S3': '3 sites', 'S4': '4 sites', 'S1_r1_coarse': '1 site\n1 µm, 5 ms'}
G = {g['name']: g for g in el['designs']}
names = [n for n in labels if n in G]

cmap = plt.get_cmap('viridis')
with plt.rc_context(RC):
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.5))
    for ax, key, lab in ((axes[0, 0], 'sd_D', r'best achievable SD of $D$ (%)'),
                         (axes[0, 1], 'sd_k', r'best achievable SD of $k$ (%)')):
        for i, dt in enumerate(A['dt_ms']):
            ax.semilogy(r, A[key][i], '-', color=cmap(i / (len(A['dt_ms']) - 1)),
                        label=f'{dt:g} ms')
        ax.axhline(m['scattered_400'][key], color='k', ls='--', lw=0.9,
                   label='400 scattered points')
        ax.set_xlabel(r'electrode distance from release site ($\mu$m)')
        ax.set_ylabel(lab); ax.grid(which='major', **GRID)
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(h, l, title='sampling interval', title_fontsize=6.5, frameon=False,
                      loc='upper left', ncol=2)
    axes[0, 1].set_ylim(top=axes[0, 0].get_ylim()[1])

    xi = np.arange(len(names))
    for ax, p, col in ((axes[1, 0], 'D', 'C3'), (axes[1, 1], 'k', 'C0')):
        b = [crb[n][f'sd_{p}'] for n in names]
        ref = [G[n]['summary'][f'ref_rmse_{p}'] for n in names]
        pin = [G[n]['summary'][f'pinn_rmse_{p}'] for n in names]
        light = 0.3 * np.array(matplotlib.colors.to_rgb(col)) + 0.7
        ax.bar(xi, b, 0.6, color=light, label='bound (SD)')
        if R50 is not None:
            ax.plot(xi - 0.2, [R50[n][f'rmse_{p}'] for n in names], '^', color='k', ms=5,
                    label='reference, 50 realizations')
        ax.plot(xi, ref, '^', color='k', mfc='white', mew=0.8, ms=5,
                label='reference, same 5 as PINN' if R50 is not None else 'reference estimator (RMSE)')
        ax.plot(xi + 0.2, pin, 'o', color=col, mec='k', mew=0.4, ms=5, label='PINN, 5 realizations')
        ax.set_yscale('log')
        ax.set_xticks(xi); ax.set_xticklabels([labels[n] for n in names], fontsize=6.5)
        ax.set_ylabel(f'error in ${p}$ (%)'); ax.grid(axis='y', which='major', **GRID)
        ax.set_ylim(2, 400)
        ax.legend(frameon=False, loc='upper center', ncol=2, fontsize=6,
                  handletextpad=0.3, columnspacing=0.8)
    for ax, ch in zip(axes.ravel(), 'abcd'):
        ax.annotate(ch, xy=(0, 1), xycoords='axes fraction', xytext=(-30, 4),
                    textcoords='offset points', fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(MS_DIR / f'Fig{NO}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'Figure_{NO}.eps', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'Figure_{NO}.tif', bbox_inches='tight', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(HERE / 'electrode_fig_preview.png', bbox_inches='tight', dpi=200)
print(f'wrote Fig{NO}.pdf, Figure_{NO}.eps/.tif')
