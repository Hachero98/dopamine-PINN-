"""PINN vs reference-estimator errors over 20 noise realizations (BSPC manuscript, Fig. 2).

Reads E1_noise_draws.json (dopamine_PINN_extensions.ipynb, experiment E1) and writes
Fig2_noise_draws_preview.png (300 dpi), .eps and 600-dpi .tif to <out_dir>
(129 mm wide, 8-pt sans-serif).

Usage:  python scripts/plot_noise_draws_fig.py [E1_noise_draws.json] [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parent / 'results' / 'manuscript_extensions' / 'E1_noise_draws.json'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')
PRINT_RC = {'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
            'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.markersize': 4.5}
MID_W = 129 / 25.4
GRID = dict(color='0.88', lw=0.5)

e1 = json.loads(SRC.read_text())
runs = [e1['runs'][str(s)] for s in e1['draw_seeds']]
A = lambda key: np.array([r[key] for r in runs])

with plt.rc_context(PRINT_RC):
    fig, axes = plt.subplots(1, 2, figsize=(MID_W, 2.45))
    for ax, q, sd_key in ((axes[0], 'D', 'ref_sd_logD'), (axes[1], 'k', 'ref_sd_logk')):
        x, y = A(f'ref_{q}_err'), A(f'pinn_{q}_err')
        lim = 1.12 * np.max(np.abs(np.concatenate([x, y])))
        band = 196 * np.mean(A(sd_key))          # mean Laplace 95% half-width, %
        ax.add_patch(plt.Rectangle((-band, -band), 2 * band, 2 * band, fill=False,
                                   ec='0.35', ls=':', lw=0.9, zorder=2))
        ax.plot([-lim, lim], [-lim, lim], color='0.4', ls='--', lw=0.8, zorder=1)
        ax.axhline(0, color='0.6', lw=0.5, zorder=1); ax.axvline(0, color='0.6', lw=0.5, zorder=1)
        ax.plot(x, y, 'o', color='C3', mec='k', mew=0.4, zorder=3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        ax.set_xlabel(f'reference estimator, error in ${q}$ (%)')
        ax.set_ylabel(f'PINN, error in ${q}$ (%)')
        ax.grid(**GRID)
    for ax, ch in zip(axes, 'ab'):
        ax.annotate(ch, xy=(0, 1), xycoords='axes fraction', xytext=(-30, 4),
                    textcoords='offset points', fontsize=9, fontweight='bold')
    fig.tight_layout()
    stem = 'Fig2_noise_draws'
    fig.savefig(OUT / f'{stem}_preview.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.eps', bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.tif', dpi=600, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
print(f'wrote {stem}_preview.png, .eps, .tif from {SRC.name}')
